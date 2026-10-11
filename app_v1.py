"""Streamlit MCP Host + Langfuse Prompt Management demo.

Architecture:

    Streamlit
        |
        v
    Langfuse Prompt Management
        |
        v
    LangChain ChatPromptTemplate
        |
        v
    LLM Tool Calling
        |
        v
    MCP Client
        |
        v
    Banking MCP Server
        |
        v
    Banking Tools

Langfuse is used for:
    1. Prompt Management
    2. Prompt Versioning
    3. LLM Observability

MCP is used for:
    1. Tool Discovery
    2. Tool Invocation
"""


import asyncio
import os

import streamlit as st
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.messages import ToolMessage
from langchain_core.prompts import ChatPromptTemplate

from mcp import Client

from langfuse import get_client, observe
from langfuse.langchain import CallbackHandler


# ---------------------------------------------------------
# Environment
# ---------------------------------------------------------

load_dotenv()


GROQ_API_KEY = os.getenv("GROQ_API_KEY")

GROQ_MODEL = os.getenv(
    "GROQ_MODEL",
    "openai/gpt-oss-20b",
)

MCP_SERVER_URL = os.getenv(
    "MCP_SERVER_URL",
    "http://127.0.0.1:8000/mcp",
)


# ---------------------------------------------------------
# Langfuse
# ---------------------------------------------------------

langfuse = get_client()

langfuse_handler = CallbackHandler()


# ---------------------------------------------------------
# Streamlit configuration
# ---------------------------------------------------------

st.set_page_config(
    page_title="Bank AI Assistant - MCP Demo",
    page_icon="🏦",
    layout="wide",
)

st.title("🏦 Bank AI Assistant — MCP Demo")

st.caption(
    "LLM Tool Calling + MCP Client + Banking MCP Server + "
    "Langfuse Prompt Management"
)


# ---------------------------------------------------------
# Validate configuration
# ---------------------------------------------------------

if not GROQ_API_KEY:

    st.error(
        "GROQ_API_KEY was not found. "
        "Add it to your .env file."
    )

    st.stop()


# ---------------------------------------------------------
# MCP: Discover tools
# ---------------------------------------------------------

async def discover_mcp_tools():
    """Discover tools exposed by the MCP server."""

    async with Client(MCP_SERVER_URL) as client:

        result = await client.list_tools()

        return [
            {
                "name": tool.name,
                "description": tool.description or "",
                "input_schema": getattr(
                    tool,
                    "inputSchema",
                    {},
                ),
            }
            for tool in result.tools
        ]


# ---------------------------------------------------------
# MCP: Invoke tool
# ---------------------------------------------------------

@observe(as_type="tool")
async def call_mcp_tool(
    tool_name: str,
    arguments: dict,
):
    """Invoke one tool through the MCP server."""

    async with Client(MCP_SERVER_URL) as client:

        return await client.call_tool(
            tool_name,
            arguments,
        )


# ---------------------------------------------------------
# 1. MCP Server
# ---------------------------------------------------------

st.subheader("1️⃣ MCP Server")

st.code(MCP_SERVER_URL)


if st.button("🔎 Discover MCP Tools"):

    try:

        with st.spinner(
            "Connecting to MCP server..."
        ):

            discovered = asyncio.run(
                discover_mcp_tools()
            )

        st.session_state["mcp_tools"] = discovered

        st.success(
            f"Discovered {len(discovered)} MCP tools."
        )

    except Exception as exc:

        st.error(
            f"Could not connect to MCP server: {exc}"
        )

        st.info(
            "Start the server first with: python server.py"
        )


# ---------------------------------------------------------
# Display discovered tools
# ---------------------------------------------------------

mcp_tools = st.session_state.get(
    "mcp_tools",
    [],
)


if mcp_tools:

    st.write(
        "**Tools exposed by the MCP server:**"
    )

    for item in mcp_tools:

        st.write(
            f"- **{item['name']}** — "
            f"{item['description']}"
        )


st.divider()


# ---------------------------------------------------------
# 2. Ask AI Assistant
# ---------------------------------------------------------

st.subheader("2️⃣ Ask the AI Assistant")


account_id = st.text_input(
    "Demo Account ID",
    value="12345",
)


question = st.text_input(
    "What would you like to know?",
    placeholder=(
        "What is my current account balance?"
    ),
)


if st.button(
    "🤖 Ask AI Assistant",
    type="primary",
):

    # -----------------------------------------------------
    # Validate user input
    # -----------------------------------------------------

    if not question.strip():

        st.warning(
            "Please enter a question."
        )

        st.stop()


    if not mcp_tools:

        st.warning(
            "Click 'Discover MCP Tools' first."
        )

        st.stop()


    # -----------------------------------------------------
    # Retrieve prompt from Langfuse
    # -----------------------------------------------------

    try:

        langfuse_prompt = langfuse.get_prompt(
            "banking-assistant",
            type="chat",
            version=2,
        )

    except Exception as exc:

        st.error(
            f"Could not retrieve Langfuse prompt: {exc}"
        )

        st.stop()


    # -----------------------------------------------------
    # Convert Langfuse prompt to LangChain prompt
    #
    # The Langfuse prompt object is attached as metadata.
    # This allows Langfuse to associate the prompt version
    # with the resulting LLM generation.
    # -----------------------------------------------------

    langchain_prompt = ChatPromptTemplate(
        langfuse_prompt.get_langchain_prompt(),
        metadata={
            "langfuse_prompt": langfuse_prompt
        },
    )


    # -----------------------------------------------------
    # Convert MCP tools into provider-neutral schemas
    # -----------------------------------------------------

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


    # -----------------------------------------------------
    # Create LLM
    # -----------------------------------------------------

    llm = ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model=GROQ_MODEL,
        temperature=0,
    )


    # -----------------------------------------------------
    # Bind MCP-discovered tools to the LLM
    # -----------------------------------------------------

    llm_with_tools = llm.bind_tools(
        tool_schemas
    )


    # -----------------------------------------------------
    # Create prompt + LLM chain
    #
    # This is important for Langfuse prompt linking.
    # -----------------------------------------------------

    prompt_chain = (
        langchain_prompt
        | llm_with_tools
    )


    # -----------------------------------------------------
    # IMPORTANT:
    #
    # Build the initial conversation messages separately.
    #
    # We need these messages later when we send the
    # MCP tool result back to the LLM.
    #
    # This fixes the conversation-state issue.
    # -----------------------------------------------------

    prompt_messages = (
        langchain_prompt.format_messages(
            question=question
        )
    )

    messages = list(prompt_messages)


    # -----------------------------------------------------
    # Tool-calling state
    # -----------------------------------------------------

    trace = []

    final_answer = None


    # -----------------------------------------------------
    # First LLM call
    #
    # This call goes through the prompt chain so that
    # Langfuse can associate the generation with:
    #
    #     banking-assistant
    #     Version 1
    #     production
    #
    # -----------------------------------------------------

    try:

        with st.spinner(
            "The AI assistant is thinking..."
        ):

            response = prompt_chain.invoke(
                {
                    "question": question
                },
                config={
                    "callbacks": [
                        langfuse_handler
                    ]
                },
            )

    except Exception as exc:

        st.error(
            f"LLM invocation failed: {exc}"
        )

        st.stop()


    # -----------------------------------------------------
    # Add the assistant's response to conversation history
    # -----------------------------------------------------

    messages.append(response)


    # -----------------------------------------------------
    # Tool-calling loop
    # -----------------------------------------------------

    for _ in range(5):

        # -------------------------------------------------
        # No tool call -> final answer
        # -------------------------------------------------

        if not response.tool_calls:

            final_answer = response.content

            break


        # -------------------------------------------------
        # Execute requested MCP tools
        # -------------------------------------------------

        for tool_call in response.tool_calls:

            tool_name = tool_call["name"]

            tool_args = dict(
                tool_call["args"]
            )

            tool_call_id = tool_call["id"]


            # -------------------------------------------------
            # Application-controlled account ID
            #
            # The LLM does not control this value.
            # -------------------------------------------------

            if tool_name in {
                "get_account_balance",
                "get_recent_transactions",
            }:

                tool_args["account_id"] = (
                    account_id
                )


            # -------------------------------------------------
            # Record MCP trace
            # -------------------------------------------------

            trace.append(
                {
                    "tool": tool_name,
                    "arguments": tool_args,
                    "via": "MCP Client",
                }
            )


            # -------------------------------------------------
            # Invoke MCP tool
            # -------------------------------------------------

            try:

                mcp_result = asyncio.run(
                    call_mcp_tool(
                        tool_name,
                        tool_args,
                    )
                )


                # -------------------------------------------------
                # Extract text from MCP result
                # -------------------------------------------------

                parts = []

                for content in getattr(
                    mcp_result,
                    "content",
                    [],
                ):

                    text_value = getattr(
                        content,
                        "text",
                        None,
                    )

                    if text_value is not None:

                        parts.append(
                            text_value
                        )


                tool_result = (
                    "\n".join(parts)
                    or str(mcp_result)
                )


            except Exception as exc:

                tool_result = (
                    f"MCP tool execution failed: "
                    f"{exc}"
                )


            # -------------------------------------------------
            # Add MCP result to conversation history
            # -------------------------------------------------

            messages.append(
                ToolMessage(
                    content=tool_result,
                    tool_call_id=tool_call_id,
                )
            )


        # -------------------------------------------------
        # Send complete conversation back to LLM
        #
        # messages now contains:
        #
        # SystemMessage
        # HumanMessage
        # AIMessage(tool call)
        # ToolMessage(result)
        #
        # -------------------------------------------------

        try:

            response = llm_with_tools.invoke(
                messages,
                config={
                    "callbacks": [
                        langfuse_handler
                    ]
                },
            )

        except Exception as exc:

            st.error(
                f"LLM invocation failed: {exc}"
            )

            st.stop()


        # -------------------------------------------------
        # Add latest assistant response to history
        # -------------------------------------------------

        messages.append(response)


    # -----------------------------------------------------
    # Tool-call limit reached
    # -----------------------------------------------------

    if final_answer is None:

        if response and not response.tool_calls:

            final_answer = response.content

        else:

            final_answer = (
                "I could not complete the request "
                "within the tool-call limit."
            )


    # ---------------------------------------------------------
    # 3. Answer
    # ---------------------------------------------------------

    st.subheader("3️⃣ Answer")

    st.markdown(final_answer)


    # ---------------------------------------------------------
    # MCP Trace
    # ---------------------------------------------------------

    st.subheader("🔍 MCP Trace")


    if trace:

        for index, item in enumerate(
            trace,
            start=1,
        ):

            st.write(
                f"**Step {index}: "
                f"{item['tool']}**"
            )

            st.json(item)

    else:

        st.info(
            "No MCP tool was required for this question."
        )


    # ---------------------------------------------------------
    # Langfuse Prompt Information
    # ---------------------------------------------------------

    st.subheader(
        "📝 Prompt Information"
    )

    st.write(
        f"**Prompt:** "
        f"{langfuse_prompt.name}"
    )

    st.write(
        f"**Version:** "
        f"{langfuse_prompt.version}"
    )

    st.write(
        "The prompt is managed and versioned "
        "in Langfuse."
    )


# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------

st.divider()

st.caption(
    "Teaching point: Tool Calling lets the LLM request "
    "an action; MCP standardizes how the AI application "
    "discovers and invokes capabilities; Langfuse manages "
    "prompt versions and provides observability."
)