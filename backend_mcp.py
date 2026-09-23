from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_groq import ChatGroq
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool, BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from dotenv import load_dotenv
import aiosqlite
import requests
import asyncio
import atexit
import threading
from retrieval import retrieve_relevant_chunks
import os

load_dotenv()

# Dedicated async loop for backend tasks
_ASYNC_LOOP = asyncio.new_event_loop()
_ASYNC_THREAD = threading.Thread(target=_ASYNC_LOOP.run_forever, daemon=True)
_ASYNC_THREAD.start()


def _submit_async(coro):
    return asyncio.run_coroutine_threadsafe(coro, _ASYNC_LOOP)


def run_async(coro):
    return _submit_async(coro).result()


def submit_async_task(coro):
    """Schedule a coroutine on the backend event loop."""
    return _submit_async(coro)


# -------------------
# 1. LLM
# -------------------

llm = ChatGroq(model="openai/gpt-oss-120b", api_key=os.getenv("GROQ_API_KEY"))  # or "meta-llama/llama-4-scout-17b-16e-instruct"

# -------------------
# 2. Tools
# -------------------
search_tool = DuckDuckGoSearchRun(region="us-en")

ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")


@tool
def get_stock_price(symbol: str) -> dict:
    """
    Fetch latest stock price for a given symbol (e.g. 'AAPL', 'TSLA')
    using Alpha Vantage.
    """
    if not ALPHA_VANTAGE_API_KEY:
        return {"error": "ALPHA_VANTAGE_API_KEY is not set in the environment."}
    url = (
        "https://www.alphavantage.co/query"
        f"?function=GLOBAL_QUOTE&symbol={symbol}&apikey={ALPHA_VANTAGE_API_KEY}"
    )
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        return r.json()
    except requests.RequestException as exc:
        return {"error": f"Failed to fetch stock price for {symbol}: {exc}"}


client = MultiServerMCPClient(
    {
        "deepwiki": {
            "transport": "streamable_http",  # try this first
            "url": "https://mcp.deepwiki.com/mcp"  # note: URL path may also differ per transport
        },

        "expense": {
            "transport": "streamable_http",  # if this fails, try "sse"
            "url": "https://splendid-gold-dingo.fastmcp.app/mcp"
        }
    }
)


def load_mcp_tools() -> list[BaseTool]:
    """Best-effort MCP tool loading — an unreachable/misconfigured server
    should not prevent the app from starting."""
    try:
        return run_async(client.get_tools())
    except Exception as exc:
        print(f"[warning] Failed to load MCP tools, continuing without them: {exc}")
        return []


mcp_tools = load_mcp_tools()

tools = [search_tool, get_stock_price, retrieve_relevant_chunks, *mcp_tools]
llm_with_tools = llm.bind_tools(tools) if tools else llm
print("MCP tools loaded:", [t.name for t in mcp_tools])
print("All tools:", [t.name for t in tools])
# -------------------
# 3. State
# -------------------
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# -------------------
# 4. Nodes
# -------------------
async def chat_node(state: ChatState):
    """LLM node that may answer or request a tool call."""
    messages = state["messages"]
    response = await llm_with_tools.ainvoke(messages)
    return {"messages": [response]}


tool_node = ToolNode(tools) if tools else None

# -------------------
# 5. Checkpointer
# -------------------
# Set POSTGRES_URI (e.g. a free Neon connection string) in production for
# persistent, multi-replica-safe chat history. Falls back to local SQLite
# (chatbot.db) automatically if POSTGRES_URI isn't set — handy for local
# dev, but note SQLite does NOT survive a redeploy/restart on most free
# hosting tiers, and doesn't work correctly across multiple replicas.

POSTGRES_URI = os.getenv("POSTGRES_URI")

_checkpointer_conn = None          # sqlite path
_postgres_cm = None                # postgres path (async context manager)


async def _init_checkpointer():
    global _checkpointer_conn, _postgres_cm

    if POSTGRES_URI:
        # Neon (and most managed Postgres) require sslmode=require if not
        # already present in the connection string.
        conn_str = POSTGRES_URI
        if "sslmode=" not in conn_str:
            sep = "&" if "?" in conn_str else "?"
            conn_str = f"{conn_str}{sep}sslmode=require"

        _postgres_cm = AsyncPostgresSaver.from_conn_string(conn_str)
        saver = await _postgres_cm.__aenter__()
        await saver.setup()  # creates the checkpoint tables on first run; no-op after
        print("[info] Using Postgres checkpointer for persistent chat history.")
        return saver

    print("[warning] POSTGRES_URI not set — falling back to local chatbot.db "
          "(SQLite). This will NOT persist across restarts on most free hosting "
          "tiers and is unsafe with multiple app replicas.")
    _checkpointer_conn = await aiosqlite.connect(database="chatbot.db")
    return AsyncSqliteSaver(_checkpointer_conn)


checkpointer = run_async(_init_checkpointer())


async def _close_checkpointer():
    if _checkpointer_conn is not None:
        await _checkpointer_conn.close()
    if _postgres_cm is not None:
        await _postgres_cm.__aexit__(None, None, None)


def _shutdown():
    try:
        run_async(_close_checkpointer())
    except Exception:
        pass


atexit.register(_shutdown)

# -------------------
# 6. Graph
# -------------------
graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_edge(START, "chat_node")

if tool_node:
    graph.add_node("tools", tool_node)
    graph.add_conditional_edges("chat_node", tools_condition)
    graph.add_edge("tools", "chat_node")
else:
    graph.add_edge("chat_node", END)

chatbot = graph.compile(checkpointer=checkpointer)

# -------------------
# 7. Helpers
# -------------------
async def _alist_threads():
    all_threads = set()
    async for checkpoint in checkpointer.alist(None):
        all_threads.add(checkpoint.config["configurable"]["thread_id"])
    return list(all_threads)


def retrieve_all_threads():
    return run_async(_alist_threads())


def get_thread_state(thread_id):
    """Synchronous wrapper around the async checkpointer's aget_state.
    AsyncSqliteSaver does not support the sync get_state/get_tuple path,
    so callers (e.g. the Streamlit frontend) must go through this instead
    of calling chatbot.get_state(...) directly."""
    config = {"configurable": {"thread_id": thread_id}}
    return run_async(chatbot.aget_state(config))
