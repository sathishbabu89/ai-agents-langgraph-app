"""Streamlit MCP Host + simple LLM demo.

The important architectural change from the original Tool Calling app is:

    Streamlit/LLM -> MCP Client -> Banking MCP Server -> Banking Tool

The LLM is still responsible for deciding whether a tool is needed.
MCP standardizes discovery and invocation of the external capabilities.
"""

import asyncio
import os

import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from mcp import Client
from langfuse.langchain import CallbackHandler
from langfuse import observe

load_dotenv()

langfuse_handler = CallbackHandler()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", "http://127.0.0.1:8000/mcp")

st.set_page_config(
    page_title="Bank AI Assistant - MCP Demo",
    page_icon="🏦",
    layout="wide",
)

st.title("🏦 Bank AI Assistant — MCP Demo")
st.caption("LLM Tool Calling + MCP Client + Banking MCP Server")

if not GROQ_API_KEY:
    st.error("GROQ_API_KEY was not found. Add it to your .env file.")
    st.stop()


async def discover_mcp_tools():
    """Discover tools from the MCP server."""
    async with Client(MCP_SERVER_URL) as client:
        result = await client.list_tools()
        return [
            {
                "name": tool.name,
                "description": tool.description or "",
                "input_schema": getattr(tool, "inputSchema", {}),
            }
            for tool in result.tools
        ]

@observe(as_type="tool")
async def call_mcp_tool(tool_name: str, arguments: dict):
    """Invoke one tool through MCP."""
    async with Client(MCP_SERVER_URL) as client:
        return await client.call_tool(tool_name, arguments)


# ---------------------------------------------------------
# MCP discovery
# ---------------------------------------------------------

st.subheader("1️⃣ MCP Server")
st.code(MCP_SERVER_URL)

if st.button("🔎 Discover MCP Tools"):
    try:
        with st.spinner("Connecting to MCP server..."):
            discovered = asyncio.run(discover_mcp_tools())
        st.session_state["mcp_tools"] = discovered
        st.success(f"Discovered {len(discovered)} MCP tools.")
    except Exception as exc:
        st.error(f"Could not connect to MCP server: {exc}")
        st.info("Start the server first with: python server.py")

mcp_tools = st.session_state.get("mcp_tools", [])

if mcp_tools:
    st.write("**Tools exposed by the MCP server:**")
    for item in mcp_tools:
        st.write(f"- **{item['name']}** — {item['description']}")

st.divider()

# ---------------------------------------------------------
# LLM + MCP tool calling
# ---------------------------------------------------------

st.subheader("2️⃣ Ask the AI Assistant")
account_id = st.text_input("Demo Account ID", value="12345")
question = st.text_input(
    "What would you like to know?",
    placeholder="What is my current account balance?",
)

if st.button("🤖 Ask AI Assistant", type="primary"):
    if not question.strip():
        st.warning("Please enter a question.")
        st.stop()

    if not mcp_tools:
        st.warning("Click 'Discover MCP Tools' first.")
        st.stop()

    # Convert MCP discovery metadata into the provider-neutral tool schema
    # expected by LangChain's bind_tools().
    tool_schemas = []
    for item in mcp_tools:
        tool_schemas.append(
            {
                "type": "function",
                "function": {
                    "name": item["name"],
                    "description": item["description"],
                    "parameters": item["input_schema"],
                },
            }
        )

    llm = ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model=GROQ_MODEL,
        temperature=0,
    )
    llm_with_tools = llm.bind_tools(tool_schemas)

    system_message = SystemMessage(
        content="""
You are a banking AI assistant.

You have access to banking capabilities exposed by an MCP server.

Use get_account_balance when the user asks for a current balance.
Use get_recent_transactions when the user asks for recent transactions.

Important rules:
- Use tools whenever appropriate.
- Do not invent account information.
- The application-provided account ID is the demo account currently being discussed.
- After receiving a tool result, provide a concise natural-language answer.
"""
    )

    messages = [system_message, HumanMessage(content=question)]
    trace = []
    final_answer = None

    with st.spinner("The AI assistant is thinking..."):
        for _ in range(5):
            response = llm_with_tools.invoke(
                messages,
                config={
                    "callbacks": [langfuse_handler]
                }
            )

            if not response.tool_calls:
                final_answer = response.content
                break

            messages.append(response)

            for tool_call in response.tool_calls:
                tool_name = tool_call["name"]
                tool_args = dict(tool_call["args"])
                tool_call_id = tool_call["id"]

                # The account ID is controlled by the application, not the LLM.
                if tool_name in {"get_account_balance", "get_recent_transactions"}:
                    tool_args["account_id"] = account_id

                trace.append({
                    "tool": tool_name,
                    "arguments": tool_args,
                    "via": "MCP Client",
                })

                try:
                    mcp_result = asyncio.run(
                        call_mcp_tool(tool_name, tool_args)
                    )

                    # MCP CallToolResult commonly exposes text blocks in .content.
                    parts = []
                    for content in getattr(mcp_result, "content", []):
                        text_value = getattr(content, "text", None)
                        if text_value is not None:
                            parts.append(text_value)

                    tool_result = "\n".join(parts) or str(mcp_result)

                except Exception as exc:
                    tool_result = f"MCP tool execution failed: {exc}"

                messages.append(
                    ToolMessage(
                        content=tool_result,
                        tool_call_id=tool_call_id,
                    )
                )

        if final_answer is None:
            final_answer = "I could not complete the request within the tool-call limit."

    st.subheader("3️⃣ Answer")
    st.markdown(final_answer)

    st.subheader("🔍 MCP Trace")
    for index, item in enumerate(trace, start=1):
        st.write(f"**Step {index}: {item['tool']}**")
        st.json(item)

st.divider()
st.caption(
    "Teaching point: Tool Calling lets the LLM request an action; MCP standardizes "
    "how the AI application discovers and invokes capabilities exposed by an MCP server."
)
