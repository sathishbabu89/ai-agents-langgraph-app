"""
Langfuse Evaluation Experiment - Prompt V1

Purpose:
    Run the banking assistant against the Langfuse evaluation
    dataset using Prompt Version 1.

Architecture:

    Langfuse Dataset
          |
          v
    experiment_v1.py
          |
          v
    Langfuse Prompt V1
          |
          v
    ChatPromptTemplate
          |
          v
    Groq LLM + Tool Calling
          |
          v
    MCP Client
          |
          v
    Banking MCP Server
          |
          v
    Banking Tool
          |
          v
    Final Answer
          |
          v
    Langfuse Experiment

Important:
    This experiment does NOT score the answers yet.

    It only runs Prompt V1 against the dataset and records
    the generated outputs.
"""


import asyncio
import os

from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.messages import ToolMessage
from langchain_core.prompts import ChatPromptTemplate

from mcp import Client

from langfuse import get_client


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

DATASET_NAME = "banking-assistant-evaluation"


# ---------------------------------------------------------
# Validate configuration
# ---------------------------------------------------------

if not GROQ_API_KEY:
    raise RuntimeError(
        "GROQ_API_KEY was not found in the .env file."
    )


# ---------------------------------------------------------
# Langfuse
# ---------------------------------------------------------

langfuse = get_client()


# ---------------------------------------------------------
# MCP: Discover tools
# ---------------------------------------------------------

async def discover_mcp_tools():
    """
    Discover tools exposed by the banking MCP server.
    """

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

async def call_mcp_tool(
    tool_name: str,
    arguments: dict,
):
    """
    Invoke a banking capability through MCP.
    """

    async with Client(MCP_SERVER_URL) as client:

        return await client.call_tool(
            tool_name,
            arguments,
        )


# ---------------------------------------------------------
# Helper: Extract MCP result text
# ---------------------------------------------------------

def extract_mcp_result(mcp_result):
    """
    Convert MCP CallToolResult content into a plain string.
    """

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

    return "\n".join(parts) or str(mcp_result)


# ---------------------------------------------------------
# Discover MCP tools once
# ---------------------------------------------------------

print()
print("=" * 70)
print("BANKING MCP - PROMPT V1 EVALUATION")
print("=" * 70)

print()
print(f"MCP Server : {MCP_SERVER_URL}")
print(f"Dataset    : {DATASET_NAME}")
print(f"Prompt     : banking-assistant")
print(f"Version    : 1")
print(f"Model      : {GROQ_MODEL}")
print()


try:

    mcp_tools = asyncio.run(
        discover_mcp_tools()
    )

except Exception as exc:

    raise RuntimeError(
        "Could not connect to the MCP server. "
        "Make sure 'python server.py' is running."
    ) from exc


print(
    f"Discovered {len(mcp_tools)} MCP tools:"
)

for tool in mcp_tools:

    print(
        f"  - {tool['name']} "
        f"-> {tool['description']}"
    )


# ---------------------------------------------------------
# Convert MCP tools into provider-neutral schemas
# ---------------------------------------------------------

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


# ---------------------------------------------------------
# Create LLM
# ---------------------------------------------------------

llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model=GROQ_MODEL,
    temperature=0,
)


# ---------------------------------------------------------
# Bind MCP tools to LLM
# ---------------------------------------------------------

llm_with_tools = llm.bind_tools(
    tool_schemas
)


# ---------------------------------------------------------
# Task function
# ---------------------------------------------------------

async def banking_task(
    *,
    item,
    **kwargs,
):
    """
    Execute one dataset item through the actual banking
    Tool Calling + MCP application flow.

    Langfuse passes a DatasetItem here.

    Expected dataset input:

        {
            "question": "What is my current account balance?"
        }
    """

    # -----------------------------------------------------
    # Read dataset input
    # -----------------------------------------------------

    dataset_input = item.input

    question = dataset_input["question"]


    # -----------------------------------------------------
    # Application-controlled account ID
    #
    # This matches the existing Streamlit application.
    # The account ID is NOT controlled by the LLM.
    # -----------------------------------------------------

    account_id = "12345"


    # -----------------------------------------------------
    # Retrieve Prompt V1 from Langfuse
    # -----------------------------------------------------

    langfuse_prompt = langfuse.get_prompt(
        "banking-assistant",
        type="chat",
        version=1,
    )


    # -----------------------------------------------------
    # Convert Langfuse prompt to LangChain prompt
    #
    # This keeps the prompt version associated with the
    # LLM generation.
    # -----------------------------------------------------

    langchain_prompt = ChatPromptTemplate(
        langfuse_prompt.get_langchain_prompt(),
        metadata={
            "langfuse_prompt": langfuse_prompt
        },
    )


    # -----------------------------------------------------
    # Create prompt -> LLM chain
    # -----------------------------------------------------

    prompt_chain = (
        langchain_prompt
        | llm_with_tools
    )


    # -----------------------------------------------------
    # Create initial conversation messages
    #
    # We preserve these messages because they are needed
    # when the MCP tool result is returned to the LLM.
    # -----------------------------------------------------

    prompt_messages = (
        langchain_prompt.format_messages(
            question=question
        )
    )

    messages = list(prompt_messages)


    # -----------------------------------------------------
    # First LLM call
    #
    # This call explicitly uses Prompt Version 1.
    # -----------------------------------------------------

    response = prompt_chain.invoke(
        {
            "question": question
        }
    )

    messages.append(response)


    # -----------------------------------------------------
    # Tool Calling Loop
    # -----------------------------------------------------

    final_answer = None


    for _ in range(5):

        # -------------------------------------------------
        # No tool call -> final answer
        # -------------------------------------------------

        if not response.tool_calls:

            final_answer = response.content

            break


        # -------------------------------------------------
        # Execute requested tools
        # -------------------------------------------------

        for tool_call in response.tool_calls:

            tool_name = tool_call["name"]

            tool_args = dict(
                tool_call["args"]
            )

            tool_call_id = tool_call["id"]


            # -------------------------------------------------
            # Application-controlled account ID
            # -------------------------------------------------

            if tool_name in {
                "get_account_balance",
                "get_recent_transactions",
            }:

                tool_args["account_id"] = (
                    account_id
                )


            # -------------------------------------------------
            # Invoke MCP
            #
            # IMPORTANT:
            #
            # banking_task() is async, therefore we use
            # await instead of asyncio.run().
            # -------------------------------------------------

            try:

                mcp_result = await call_mcp_tool(
                    tool_name,
                    tool_args,
                )

                tool_result = extract_mcp_result(
                    mcp_result
                )

            except Exception as exc:

                tool_result = (
                    f"MCP tool execution failed: {exc}"
                )


            # -------------------------------------------------
            # Add MCP result to conversation
            # -------------------------------------------------

            messages.append(
                ToolMessage(
                    content=tool_result,
                    tool_call_id=tool_call_id,
                )
            )


        # -------------------------------------------------
        # Send complete conversation back to LLM
        # -------------------------------------------------

        response = llm_with_tools.invoke(
            messages
        )

        messages.append(response)


    # -----------------------------------------------------
    # Handle tool-call limit
    # -----------------------------------------------------

    if final_answer is None:

        if response and not response.tool_calls:

            final_answer = response.content

        else:

            final_answer = (
                "I could not complete the request "
                "within the tool-call limit."
            )


    return final_answer


# ---------------------------------------------------------
# Run Langfuse Experiment
# ---------------------------------------------------------

print()
print("-" * 70)
print("Starting Prompt V1 experiment...")
print("-" * 70)
print()


dataset = langfuse.get_dataset(
    DATASET_NAME
)


result = dataset.run_experiment(
    name="banking-assistant-prompt-v1",
    description=(
        "Run banking assistant evaluation dataset "
        "using banking-assistant Prompt Version 1 "
        "with the real MCP tool-calling flow."
    ),
    task=banking_task,
    max_concurrency=1,
)


# ---------------------------------------------------------
# Print experiment summary
# ---------------------------------------------------------

print()
print("=" * 70)
print("EXPERIMENT COMPLETED")
print("=" * 70)

print()

print(
    result.format(
        include_item_results=True
    )
)


# ---------------------------------------------------------
# Langfuse experiment URL
# ---------------------------------------------------------

if result.dataset_run_url:

    print()
    print("View the experiment in Langfuse:")
    print(result.dataset_run_url)


# ---------------------------------------------------------
# Flush pending telemetry
# ---------------------------------------------------------

langfuse.flush()

print()
print("Langfuse telemetry flushed.")
print()