

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
from dataclasses import dataclass, field
from typing import Any, Literal, TypedDict
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END
import re
from datetime import date
from decimal import Decimal, InvalidOperation

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



@dataclass
class AgentState:
    """Tracks the agent's progress toward completing a task."""

    goal: str

    plan: list[str] = field(default_factory=list)

    observations: list[dict[str, Any]] = field(
        default_factory=list
    )

    transactions: list[dict[str, Any]] = field(
        default_factory=list
    )

    analysis: dict[str, Any] = field(
        default_factory=dict
    )

    completed_steps: list[str] = field(
        default_factory=list
    )

    missing_information: list[str] = field(
        default_factory=list
    )

    actions: list[dict[str, Any]] = field(
        default_factory=list
    )

    status: str = "initialized"

    final_answer: str = ""


def create_plan(goal: str) -> list[str]:
    """
    Create a plan for the transaction-analysis task.

    This first version uses predefined steps.
    Later, an LLM can generate and revise the plan.
    """

    return [
        "Retrieve recent transactions through MCP",
        "Validate the retrieved transaction data",
        "Calculate total expenses using Python",
        "Identify the largest expense",
        "Compare expenses against the spending limit",
        "Check whether the available data is sufficient",
        "Prepare the final response",
    ]


def initialize_agent(goal: str) -> AgentState:
    """Initialize the state for a new agent task."""

    state = AgentState(goal=goal)
    state.plan = create_plan(goal)
    state.status = "planned"

    state.actions.append({
        "action": "create_plan",
        "status": "completed",
        "details": f"Created {len(state.plan)} planned steps",
    })

    return state


def parse_transactions(raw_text: str) -> list[dict]:
    """Convert MCP's pipe-delimited text into transaction records."""

    transactions = []

    for line in raw_text.splitlines():
        line = line.strip()

        if not line:
            continue

        parts = [part.strip() for part in line.split("|")]

        if len(parts) != 3:
            raise ValueError(
                f"Unexpected transaction format: {line}"
            )

        date_text, description, amount_text = parts

        try:
            transaction_date = date.fromisoformat(date_text)

            # Remove the currency symbol and thousands separators.
            cleaned_amount = (
                amount_text
                .replace("£", "")
                .replace(",", "")
                .strip()
            )

            amount = Decimal(cleaned_amount)

        except (ValueError, InvalidOperation) as exc:
            raise ValueError(
                f"Invalid transaction data: {line}"
            ) from exc

        if not description:
            raise ValueError(
                f"Missing transaction description: {line}"
            )

        transactions.append({
            "date": transaction_date.isoformat(),
            "description": description,
            "amount": amount,
        })

    if not transactions:
        raise ValueError(
            "No valid transaction records were returned."
        )

    return transactions


def analyze_transactions(
    transactions: list[dict],
    spending_limit: Decimal,
) -> dict:
    """Calculate expense totals and compare them with a limit."""

    expenses = [
        transaction
        for transaction in transactions
        if transaction["amount"] < 0
    ]

    if not expenses:
        raise ValueError(
            "No expense transactions are available to analyze."
        )

    total_expenses = sum(
        (-transaction["amount"] for transaction in expenses),
        Decimal("0.00"),
    )

    largest_expense = max(
        expenses,
        key=lambda transaction: abs(transaction["amount"]),
    )

    return {
        "transaction_count": len(transactions),
        "expense_count": len(expenses),
        "total_expenses": total_expenses,
        "largest_expense": {
            "date": largest_expense["date"],
            "description": largest_expense["description"],
            "amount": abs(largest_expense["amount"]),
        },
        "spending_limit": spending_limit,
        "exceeded_limit": total_expenses > spending_limit,
        "amount_over_limit": max(
            Decimal("0.00"),
            total_expenses - spending_limit,
        ),
    }


def validate_transaction_data(
    transactions: list[dict],
) -> list[str]:
    """Return issues that prevent reliable analysis."""

    issues = []

    if not transactions:
        issues.append("No transactions were returned.")
        return issues

    for transaction in transactions:
        if not all(
            key in transaction
            for key in ("date", "description", "amount")
        ):
            issues.append(
                "One or more transactions are missing required fields."
            )
            break

    return issues


async def fetch_recent_transactions(
    account_id: str,
    number_of_transactions: int = 5,
) -> str:
    """Retrieve recent transactions through the MCP client."""

    result = await call_mcp_tool(
        "get_recent_transactions",
        {
            "account_id": account_id,
            "number_of_transactions": number_of_transactions,
        },
    )

    parts = []

    for content in getattr(result, "content", []):
        text_value = getattr(content, "text", None)

        if text_value is not None:
            parts.append(text_value)

    if getattr(result, "isError", False):
        raise RuntimeError(
            "\n".join(parts) or "MCP tool execution failed."
        )

    if not parts:
        raise ValueError("The MCP tool returned no transaction text.")

    return "\n".join(parts)


async def execute_transaction_analysis(
    state: AgentState,
    account_id: str,
    spending_limit: Decimal = Decimal("200.00"),
) -> AgentState:
    """Execute the planned transaction-analysis task."""

    state.status = "running"

    # Step 1: Retrieve transaction data.
    state.actions.append({
        "action": "retrieve_transactions",
        "status": "started",
    })

    try:
        raw_text = await fetch_recent_transactions(
            account_id=account_id,
            number_of_transactions=5,
        )

        state.observations.append({
            "source": "get_recent_transactions",
            "result": raw_text,
        })

        transactions = parse_transactions(raw_text)
        state.transactions = transactions

        state.actions.append({
            "action": "retrieve_transactions",
            "status": "completed",
            "transaction_count": len(transactions),
        })

        state.completed_steps.append(
            "Retrieve recent transactions through MCP"
        )

    except Exception as exc:
        state.status = "failed"
        state.missing_information.append(
            f"Could not retrieve or parse transactions: {exc}"
        )
        state.actions.append({
            "action": "retrieve_transactions",
            "status": "failed",
            "error": str(exc),
        })
        return state

    # Step 2: Validate the parsed records.
    issues = validate_transaction_data(state.transactions)

    if issues:
        state.status = "needs_attention"
        state.missing_information.extend(issues)
        return state

    state.completed_steps.append(
        "Validate the retrieved transaction data"
    )

    # Step 3: Calculate the results deterministically.
    try:
        state.analysis = analyze_transactions(
            state.transactions,
            spending_limit,
        )

        state.completed_steps.extend([
            "Calculate total expenses using Python",
            "Identify the largest expense",
            "Compare expenses against the spending limit",
        ])

    except ValueError as exc:
        state.status = "needs_attention"
        state.missing_information.append(str(exc))
        return state

    # The current MCP tool returns only a limited set of recent
    # transactions, not a complete period-based spending history.
    state.missing_information.append(
        "Only the five most recent transactions were analyzed. "
        "A complete spending total for a specific period cannot "
        "be confirmed from this response alone."
    )

    state.completed_steps.append(
        "Check whether the available data is sufficient"
    )

    state.status = "completed"
    return state


def finalize_transaction_analysis(
    state: AgentState,
) -> AgentState:
    """Create the final answer and complete the task state."""

    if not state.analysis:
        state.status = "needs_attention"
        state.final_answer = (
            "I couldn't complete the analysis because "
            "valid transaction results are unavailable."
        )
        return state

    analysis = state.analysis
    largest = analysis["largest_expense"]

    limit_result = (
        f"The returned transactions exceed the spending limit "
        f"by £{analysis['amount_over_limit']:,.2f}."
        if analysis["exceeded_limit"]
        else "The returned transactions do not exceed "
             "the spending limit."
    )

    state.final_answer = (
        f"**Total expenses:** "
        f"£{analysis['total_expenses']:,.2f}\n\n"
        f"**Largest expense:** {largest['description']} "
        f"on {largest['date']} "
        f"(£{largest['amount']:,.2f}).\n\n"
        f"**Spending limit:** "
        f"£{analysis['spending_limit']:,.2f}. "
        f"{limit_result}\n\n"
        f"**Scope:** This analysis covers "
        f"{analysis['transaction_count']} returned transactions, "
        "not necessarily all spending for a specific period."
    )

    final_step = "Prepare the final response"

    if final_step not in state.completed_steps:
        state.completed_steps.append(final_step)

    state.actions.append({
        "action": "finalize_response",
        "status": "completed",
    })

    state.status = "completed"
    return state


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
# LangGraph Agent: state, decisions, actions, and routing
# ---------------------------------------------------------

MAX_AGENT_STEPS = 6


class GraphAgentState(TypedDict, total=False):
    """Shared state passed between LangGraph nodes."""

    goal: str
    account_id: str
    spending_limit: Decimal
    transactions: list[dict]
    analysis: dict[str, Any]
    balance_result: str
    observations: list[dict[str, Any]]
    actions: list[dict[str, Any]]
    next_action: str
    decision_reason: str
    step_count: int
    status: str
    errors: list[str]
    final_answer: str


class AgentDecision(BaseModel):
    """A bounded action choice made by the LLM."""

    action: Literal[
        "get_account_balance",
        "get_recent_transactions",
        "analyze_transactions",
        "finish",
    ] = Field(description="The single next action to perform")
    reason: str = Field(description="Brief explanation for choosing this action")


def _successful_action_names(state: GraphAgentState) -> set[str]:
    return {
        item.get("action", "")
        for item in state.get("actions", [])
        if item.get("status") == "completed"
    }


def _available_agent_actions(state: GraphAgentState) -> list[str]:
    """Return actions that are currently valid, preventing repeat calls."""
    completed = _successful_action_names(state)
    available = []

    if "get_account_balance" not in completed:
        available.append("get_account_balance")
    if "get_recent_transactions" not in completed:
        available.append("get_recent_transactions")
    if state.get("transactions") and not state.get("analysis"):
        available.append("analyze_transactions")
    available.append("finish")
    return available


def _extract_mcp_text(result: Any) -> str:
    parts = []
    for content in getattr(result, "content", []):
        text_value = getattr(content, "text", None)
        if text_value is not None:
            parts.append(text_value)
    if getattr(result, "isError", False):
        raise RuntimeError("\n".join(parts) or "MCP tool execution failed.")
    if not parts:
        raise ValueError("The MCP tool returned no text content.")
    return "\n".join(parts)


def decide_next_action(state: GraphAgentState) -> dict:
    """Ask the LLM to select one currently permitted next action."""
    if state.get("step_count", 0) >= MAX_AGENT_STEPS:
        return {
            "next_action": "finish",
            "decision_reason": "Reached the maximum agent-step limit.",
        }

    available = _available_agent_actions(state)
    recent_observations = state.get("observations", [])[-6:]
    compact_analysis = state.get("analysis", {})

    decision_llm = ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model=GROQ_MODEL,
        temperature=0,
    ).with_structured_output(
        AgentDecision,
        method="json_schema",
        strict=True,
    )

    system_message = (
        "You are the decision component of a bounded banking analysis agent. "
        "Choose exactly one action from the supplied available-action list. "
        "Use the user's goal and existing observations to decide what is still "
        "needed. Retrieve balance only when relevant to the goal. Retrieve "
        "transactions when the goal asks about transactions, spending, expenses, "
        "or a spending limit. Run analyze_transactions only after transaction "
        "data is available. Choose finish when the goal has been answered, "
        "the necessary information cannot be obtained, or no useful action remains. "
        "Never invent tool results or financial calculations."
    )
    user_message = (
        f"User goal: {state.get('goal', '')}\n"
        f"Available actions: {available}\n"
        f"Account ID: {state.get('account_id', '')}\n"
        f"Spending limit: £{state.get('spending_limit', Decimal('200.00'))}\n"
        f"Balance result already obtained: {bool(state.get('balance_result'))}\n"
        f"Transaction data available: {bool(state.get('transactions'))}\n"
        f"Analysis available: {bool(compact_analysis)}\n"
        f"Observations so far: {recent_observations}\n"
        f"Errors so far: {state.get('errors', [])}\n"
        "Select one action from the available-action list and explain briefly why."
    )

    decision = decision_llm.invoke(
        [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ],
        config={"callbacks": [langfuse_handler]},
    )

    selected_action = decision.action
    if selected_action not in available:
        selected_action = "finish"
        reason = (
            f"The model selected an unavailable action. Stopping safely. "
            f"Model reason: {decision.reason}"
        )
    else:
        reason = decision.reason

    return {
        "next_action": selected_action,
        "decision_reason": reason,
        "status": "deciding",
    }


def execute_selected_action(state: GraphAgentState) -> dict:
    """Execute one LLM-selected action using MCP or deterministic Python."""
    action = state.get("next_action", "finish")
    actions = list(state.get("actions", []))
    observations = list(state.get("observations", []))
    transactions = list(state.get("transactions", []))
    analysis = dict(state.get("analysis", {}))
    errors = list(state.get("errors", []))
    updates: dict[str, Any] = {
        "actions": actions,
        "observations": observations,
        "transactions": transactions,
        "analysis": analysis,
        "errors": errors,
        "step_count": state.get("step_count", 0) + 1,
        "status": "running",
    }

    action_record: dict[str, Any] = {
        "action": action,
        "status": "started",
        "reason": state.get("decision_reason", ""),
    }
    actions.append(action_record)

    try:
        if action == "get_account_balance":
            result = asyncio.run(
                call_mcp_tool(
                    "get_account_balance",
                    {"account_id": state["account_id"]},
                )
            )
            text_result = _extract_mcp_text(result)
            updates["balance_result"] = text_result
            observations.append({"source": action, "result": text_result})
            action_record.update({"status": "completed"})

        elif action == "get_recent_transactions":
            result = asyncio.run(
                call_mcp_tool(
                    "get_recent_transactions",
                    {
                        "account_id": state["account_id"],
                        "number_of_transactions": 5,
                    },
                )
            )
            text_result = _extract_mcp_text(result)
            parsed = parse_transactions(text_result)
            issues = validate_transaction_data(parsed)
            if issues:
                raise ValueError("; ".join(issues))
            updates["transactions"] = parsed
            observations.append({"source": action, "result": text_result})
            action_record.update({
                "status": "completed",
                "transaction_count": len(parsed),
            })

        elif action == "analyze_transactions":
            current_transactions = state.get("transactions", [])
            if not current_transactions:
                raise ValueError("Cannot analyze before retrieving transactions.")
            result = analyze_transactions(
                current_transactions,
                state.get("spending_limit", Decimal("200.00")),
            )
            updates["analysis"] = result
            observations.append({
                "source": action,
                "result": {
                    "total_expenses": str(result["total_expenses"]),
                    "largest_expense": result["largest_expense"],
                    "exceeded_limit": result["exceeded_limit"],
                },
            })
            action_record.update({"status": "completed"})

        else:
            action_record.update({"status": "skipped"})

    except Exception as exc:
        error_message = f"{action} failed: {exc}"
        errors.append(error_message)
        observations.append({"source": action, "error": str(exc)})
        action_record.update({"status": "failed", "error": str(exc)})

    updates["actions"] = actions
    updates["observations"] = observations
    updates["errors"] = errors
    return updates


def route_after_decision(state: GraphAgentState) -> str:
    if state.get("next_action") == "finish":
        return "finalize"
    if state.get("step_count", 0) >= MAX_AGENT_STEPS:
        return "finalize"
    return "execute_action"


def finalize_agent_result(state: GraphAgentState) -> dict:
    """Create a user-facing answer from observed tool and Python results."""
    parts = []
    goal = state.get("goal", "")
    balance_result = state.get("balance_result", "")
    analysis = state.get("analysis", {})
    transactions = state.get("transactions", [])
    errors = state.get("errors", [])

    if balance_result and any(
        word in goal.lower() for word in ("balance", "account", "money")
    ):
        parts.append(f"**Account balance:** {balance_result}")

    if analysis:
        largest = analysis["largest_expense"]
        if analysis["exceeded_limit"]:
            limit_summary = (
                f"The returned expenses exceed the spending limit by "
                f"£{analysis['amount_over_limit']:,.2f}."
            )
        else:
            remaining = analysis["spending_limit"] - analysis["total_expenses"]
            limit_summary = (
                f"The returned expenses do not exceed the spending limit. "
                f"They are £{remaining:,.2f} below it."
            )
        parts.append(
            f"**Total expenses:** £{analysis['total_expenses']:,.2f}\n\n"
            f"**Largest expense:** {largest['description']} on "
            f"{largest['date']} (£{largest['amount']:,.2f}).\n\n"
            f"**Spending limit:** £{analysis['spending_limit']:,.2f}. "
            f"{limit_summary}\n\n"
            f"**Scope:** This analysis covers {analysis['transaction_count']} "
            "returned transactions, not necessarily all spending for a specific period."
        )
    elif transactions:
        parts.append(
            f"Retrieved {len(transactions)} transactions. The goal did not require "
            "a financial calculation, or the analysis could not be completed."
        )

    if errors:
        parts.append("**Issues encountered:**\n" + "\n".join(f"- {e}" for e in errors))
    if not parts:
        parts.append(
            "I couldn't complete the request with the information available. "
            "Please check the goal, account ID, and MCP server status."
        )

    status = "completed" if (balance_result or analysis or transactions) and not errors else "needs_attention"
    if state.get("step_count", 0) >= MAX_AGENT_STEPS:
        parts.append(
            f"**Safety limit:** The agent reached its maximum of {MAX_AGENT_STEPS} action steps."
        )

    return {"final_answer": "\n\n".join(parts), "status": status}


def build_agent_graph():
    """Build a bounded decide -> act -> observe -> decide LangGraph loop."""
    graph = StateGraph(GraphAgentState)
    graph.add_node("decide", decide_next_action)
    graph.add_node("execute_action", execute_selected_action)
    graph.add_node("finalize", finalize_agent_result)
    graph.add_edge(START, "decide")
    graph.add_conditional_edges(
        "decide",
        route_after_decision,
        {
            "execute_action": "execute_action",
            "finalize": "finalize",
        },
    )
    graph.add_edge("execute_action", "decide")
    graph.add_edge("finalize", END)
    return graph.compile()


agent_graph = build_agent_graph()



# ---------------------------------------------------------
# LangGraph Multi-Agent Workflow: supervisor + specialist agents
# ---------------------------------------------------------

MAX_MULTIAGENT_STEPS = 6


class MultiAgentState(TypedDict, total=False):
    """Shared state passed between the supervisor and specialist agents."""

    goal: str
    account_id: str
    spending_limit: Decimal
    balance_result: str
    transactions: list[dict]
    analysis: dict[str, Any]
    specialist_reports: dict[str, str]
    completed_agents: list[str]
    next_agent: str
    supervisor_reason: str
    agent_trace: list[dict[str, Any]]
    step_count: int
    errors: list[str]
    final_answer: str
    status: str


class SupervisorDecision(BaseModel):
    """Supervisor chooses which specialist should work next."""

    next_agent: Literal[
        "balance_agent",
        "transaction_agent",
        "budget_agent",
        "response_agent",
    ] = Field(description="The next specialist agent to invoke")
    reason: str = Field(description="Why this specialist is the best next step")


def _multiagent_llm():
    return ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model=GROQ_MODEL,
        temperature=0,
    )


def _multiagent_goal_needs_balance(goal: str) -> bool:
    lowered = goal.lower()
    return any(term in lowered for term in ("balance", "available funds", "funds in my account"))


def _multiagent_goal_needs_transactions(goal: str) -> bool:
    lowered = goal.lower()
    return any(term in lowered for term in (
        "transaction", "spending", "expense", "expenses", "budget",
        "limit", "largest", "merchant", "purchase", "spent", "spend",
    ))


def _multiagent_goal_needs_budget_analysis(goal: str) -> bool:
    lowered = goal.lower()
    return any(term in lowered for term in (
        "spending", "expense", "expenses", "budget", "limit", "largest",
        "total", "analy", "spent", "spend",
    ))


def _multiagent_available_agents(state: MultiAgentState) -> list[str]:
    """Offer only relevant specialists that have not already completed."""
    goal = state.get("goal", "")
    completed = set(state.get("completed_agents", []))
    available: list[str] = []

    if _multiagent_goal_needs_balance(goal) and "balance_agent" not in completed:
        available.append("balance_agent")

    if (
        _multiagent_goal_needs_transactions(goal)
        and "transaction_agent" not in completed
    ):
        available.append("transaction_agent")

    if (
        _multiagent_goal_needs_budget_analysis(goal)
        and state.get("transactions")
        and "budget_agent" not in completed
    ):
        available.append("budget_agent")

    # A balance request can be the only requested task; transaction requests
    # are only offered when the goal suggests transaction-related work.
    available.append("response_agent")
    return available


def _multiagent_record(state: MultiAgentState, record: dict[str, Any]) -> list[dict[str, Any]]:
    trace = list(state.get("agent_trace", []))
    trace.append(record)
    return trace


def multiagent_supervisor(state: MultiAgentState) -> dict:
    """Supervisor agent uses an LLM to route work to a specialist."""
    completed = list(state.get("completed_agents", []))
    trace = list(state.get("agent_trace", []))
    if state.get("step_count", 0) >= MAX_MULTIAGENT_STEPS:
        return {
            "next_agent": "response_agent",
            "supervisor_reason": "Reached the configured multi-agent step limit; finalize with available evidence.",
            "agent_trace": trace + [{
                "role": "supervisor",
                "decision": "response_agent",
                "reason": "Maximum step limit reached",
            }],
        }

    available = _multiagent_available_agents(state)
    supervisor_llm = _multiagent_llm().with_structured_output(
        SupervisorDecision,
        method="json_schema",
        strict=True,
    )
    system_message = (
        "You are the supervisor of a banking multi-agent workflow. Route the request "
        "to exactly one specialist from the available list. The balance specialist "
        "retrieves account balance; the transaction specialist retrieves and validates "
        "transactions; the budget specialist calculates spending totals, largest expense, "
        "and limit comparison using Python results; the response specialist combines "
        "available results into the final answer. Choose only from available agents. "
        "Do not send work to a specialist that has already completed. Prefer the response "
        "specialist only when all relevant available work is done or no other relevant "
        "specialist is available. Never invent financial facts."
    )
    user_message = (
        f"User goal: {state.get('goal', '')}\n"
        f"Available specialist agents: {available}\n"
        f"Completed agents: {completed}\n"
        f"Balance retrieved: {bool(state.get('balance_result'))}\n"
        f"Transactions retrieved: {bool(state.get('transactions'))}\n"
        f"Budget analysis available: {bool(state.get('analysis'))}\n"
        f"Specialist reports so far: {state.get('specialist_reports', {})}\n"
        f"Errors: {state.get('errors', [])}\n"
        "Choose the next specialist and explain why."
    )
    decision = supervisor_llm.invoke(
        [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ],
        config={"callbacks": [langfuse_handler]},
    )
    chosen = decision.next_agent
    reason = decision.reason
    if chosen not in available:
        chosen = "response_agent"
        reason = "Supervisor selected an unavailable specialist; routing safely to response generation."
    elif chosen == "response_agent":
        # Guard against premature finalization: if relevant specialists remain,
        # complete the next available specialist before composing the final answer.
        pending_specialists = [name for name in available if name != "response_agent"]
        if pending_specialists:
            chosen = pending_specialists[0]
            reason = (
                "The supervisor proposed finalization, but the workflow guard found "
                f"relevant work still pending. Running {chosen} first. Original rationale: {decision.reason}"
            )

    return {
        "next_agent": chosen,
        "supervisor_reason": reason,
        "agent_trace": trace + [{
            "role": "supervisor",
            "decision": chosen,
            "reason": reason,
            "available_agents": available,
        }],
    }


def multiagent_balance_agent(state: MultiAgentState) -> dict:
    """Balance specialist retrieves balance and produces a focused finding."""
    trace = list(state.get("agent_trace", []))
    completed = list(state.get("completed_agents", []))
    reports = dict(state.get("specialist_reports", {}))
    errors = list(state.get("errors", []))
    balance_result = state.get("balance_result", "")
    try:
        result = asyncio.run(call_mcp_tool(
            "get_account_balance", {"account_id": state["account_id"]}
        ))
        balance_result = _extract_mcp_text(result)
        prompt = (
            "You are the Balance Specialist in a banking support system. "
            "Summarize only the balance result supplied below. Do not infer affordability "
            "or invent facts. Keep it to one or two sentences.\n\n"
            f"User goal: {state.get('goal', '')}\nBalance result: {balance_result}"
        )
        response = _multiagent_llm().invoke(
            prompt, config={"callbacks": [langfuse_handler]}
        )
        reports["balance_agent"] = str(response.content)
        trace.append({"role": "balance_agent", "status": "completed", "finding": reports["balance_agent"]})
    except Exception as exc:
        message = f"Balance Agent failed: {exc}"
        errors.append(message)
        reports["balance_agent"] = message
        trace.append({"role": "balance_agent", "status": "failed", "error": str(exc)})
    if "balance_agent" not in completed:
        completed.append("balance_agent")
    return {
        "balance_result": balance_result,
        "specialist_reports": reports,
        "completed_agents": completed,
        "agent_trace": trace,
        "errors": errors,
        "step_count": state.get("step_count", 0) + 1,
    }


def multiagent_transaction_agent(state: MultiAgentState) -> dict:
    """Transaction specialist retrieves and validates transaction records."""
    trace = list(state.get("agent_trace", []))
    completed = list(state.get("completed_agents", []))
    reports = dict(state.get("specialist_reports", {}))
    errors = list(state.get("errors", []))
    transactions = list(state.get("transactions", []))
    try:
        result = asyncio.run(call_mcp_tool(
            "get_recent_transactions",
            {"account_id": state["account_id"], "number_of_transactions": 5},
        ))
        raw_text = _extract_mcp_text(result)
        parsed = parse_transactions(raw_text)
        issues = validate_transaction_data(parsed)
        if issues:
            raise ValueError("; ".join(issues))
        transactions = parsed
        prompt = (
            "You are the Transaction Specialist in a banking support system. "
            "Summarize the returned transaction data without calculating a total or "
            "claiming it represents all account activity. Mention the number of returned "
            "transactions and a few merchants if useful.\n\n"
            f"User goal: {state.get('goal', '')}\nRaw transaction data:\n{raw_text}"
        )
        response = _multiagent_llm().invoke(
            prompt, config={"callbacks": [langfuse_handler]}
        )
        reports["transaction_agent"] = str(response.content)
        trace.append({
            "role": "transaction_agent", "status": "completed",
            "transaction_count": len(transactions), "finding": reports["transaction_agent"],
        })
    except Exception as exc:
        message = f"Transaction Agent failed: {exc}"
        errors.append(message)
        reports["transaction_agent"] = message
        trace.append({"role": "transaction_agent", "status": "failed", "error": str(exc)})
    if "transaction_agent" not in completed:
        completed.append("transaction_agent")
    return {
        "transactions": transactions,
        "specialist_reports": reports,
        "completed_agents": completed,
        "agent_trace": trace,
        "errors": errors,
        "step_count": state.get("step_count", 0) + 1,
    }


def multiagent_budget_agent(state: MultiAgentState) -> dict:
    """Budget specialist uses exact Python arithmetic, then explains the result."""
    trace = list(state.get("agent_trace", []))
    completed = list(state.get("completed_agents", []))
    reports = dict(state.get("specialist_reports", {}))
    errors = list(state.get("errors", []))
    analysis = dict(state.get("analysis", {}))
    try:
        transactions = state.get("transactions", [])
        if not transactions:
            raise ValueError("No transaction data is available for budget analysis.")
        analysis = analyze_transactions(
            transactions,
            state.get("spending_limit", Decimal("200.00")),
        )
        facts = (
            f"Returned transaction count: {analysis['transaction_count']}\n"
            f"Total expenses: £{analysis['total_expenses']:.2f}\n"
            f"Largest expense: {analysis['largest_expense']['description']} "
            f"on {analysis['largest_expense']['date']}, "
            f"£{analysis['largest_expense']['amount']:.2f}\n"
            f"Spending limit: £{analysis['spending_limit']:.2f}\n"
            f"Exceeded limit: {analysis['exceeded_limit']}\n"
            f"Amount over limit: £{analysis['amount_over_limit']:.2f}"
        )
        prompt = (
            "You are the Budget Analysis Specialist. Python has already performed the "
            "financial calculations. Explain these supplied facts clearly; do not change "
            "the numbers or perform unsupported calculations. State that the analysis is "
            "limited to the returned transactions.\n\n"
            f"User goal: {state.get('goal', '')}\nCalculated facts:\n{facts}"
        )
        response = _multiagent_llm().invoke(
            prompt, config={"callbacks": [langfuse_handler]}
        )
        reports["budget_agent"] = str(response.content)
        trace.append({"role": "budget_agent", "status": "completed", "calculated_facts": facts, "finding": reports["budget_agent"]})
    except Exception as exc:
        message = f"Budget Agent failed: {exc}"
        errors.append(message)
        reports["budget_agent"] = message
        trace.append({"role": "budget_agent", "status": "failed", "error": str(exc)})
    if "budget_agent" not in completed:
        completed.append("budget_agent")
    return {
        "analysis": analysis,
        "specialist_reports": reports,
        "completed_agents": completed,
        "agent_trace": trace,
        "errors": errors,
        "step_count": state.get("step_count", 0) + 1,
    }


def multiagent_response_agent(state: MultiAgentState) -> dict:
    """Response agent combines specialist findings into the final answer."""
    reports = state.get("specialist_reports", {})
    errors = state.get("errors", [])
    prompt = (
        "You are the Response Agent in a banking multi-agent workflow. Compose a clear, "
        "concise answer to the user's goal using only the specialist reports and exact "
        "calculated facts provided. Do not invent missing information. If the reports "
        "show that only five transactions were returned, clearly state the scope limitation. "
        "If a specialist failed, be transparent about what could not be completed.\n\n"
        f"User goal: {state.get('goal', '')}\n"
        f"Balance result: {state.get('balance_result', '')}\n"
        f"Calculated analysis: {state.get('analysis', {})}\n"
        f"Specialist reports: {reports}\n"
        f"Errors: {errors}"
    )
    try:
        response = _multiagent_llm().invoke(
            prompt, config={"callbacks": [langfuse_handler]}
        )
        answer = str(response.content)
    except Exception as exc:
        # Keep a useful deterministic fallback if response generation fails.
        pieces = []
        if state.get("balance_result"):
            pieces.append(state["balance_result"])
        analysis = state.get("analysis", {})
        if analysis:
            largest = analysis["largest_expense"]
            pieces.append(
                f"Total expenses: £{analysis['total_expenses']:.2f}. "
                f"Largest expense: {largest['description']} on {largest['date']} "
                f"(£{largest['amount']:.2f}). Spending limit: £{analysis['spending_limit']:.2f}. "
                + (f"Exceeded by £{analysis['amount_over_limit']:.2f}." if analysis["exceeded_limit"] else "The limit was not exceeded.")
            )
        pieces.extend(errors)
        pieces.append(f"Response Agent could not generate an LLM response: {exc}")
        answer = "\n\n".join(pieces) or "I couldn't complete the request with the available information."

    trace = list(state.get("agent_trace", []))
    trace.append({"role": "response_agent", "status": "completed", "summary": "Combined specialist findings into the final response."})
    return {
        "final_answer": answer,
        "agent_trace": trace,
        "status": "completed" if not errors else "needs_attention",
    }


def route_multiagent_supervisor(state: MultiAgentState) -> str:
    """Route supervisor's choice to the appropriate specialist node."""
    target = state.get("next_agent", "response_agent")
    allowed = {"balance_agent", "transaction_agent", "budget_agent", "response_agent"}
    return target if target in allowed else "response_agent"


def build_multiagent_graph():
    """Build a supervisor-led graph with three specialist agents and a response agent."""
    graph = StateGraph(MultiAgentState)
    graph.add_node("supervisor", multiagent_supervisor)
    graph.add_node("balance_agent", multiagent_balance_agent)
    graph.add_node("transaction_agent", multiagent_transaction_agent)
    graph.add_node("budget_agent", multiagent_budget_agent)
    graph.add_node("response_agent", multiagent_response_agent)
    graph.add_edge(START, "supervisor")
    graph.add_conditional_edges(
        "supervisor",
        route_multiagent_supervisor,
        {
            "balance_agent": "balance_agent",
            "transaction_agent": "transaction_agent",
            "budget_agent": "budget_agent",
            "response_agent": "response_agent",
        },
    )
    graph.add_edge("balance_agent", "supervisor")
    graph.add_edge("transaction_agent", "supervisor")
    graph.add_edge("budget_agent", "supervisor")
    graph.add_edge("response_agent", END)
    return graph.compile()


multiagent_graph = build_multiagent_graph()


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
# LangGraph Agent UI
# ---------------------------------------------------------

st.divider()
st.header("3️⃣ Transaction Analysis Agent — LangGraph")
st.caption(
    "An adaptive, bounded workflow: the LLM chooses the next permitted action; "
    "MCP retrieves banking data; Python performs financial calculations."
)

account_id_agent = st.text_input(
    "Account ID",
    value="12345",
    key="agent_account_id",
)

spending_limit_input = st.number_input(
    "Spending limit (£)",
    min_value=0.0,
    value=200.0,
    step=10.0,
    key="agent_spending_limit",
)

agent_goal = st.text_area(
    "Agent goal",
    value=(
        "Check my current account balance, analyze my recent transactions, "
        "calculate my spending total, identify my largest expense, and tell "
        "me whether I exceeded my spending limit."
    ),
    key="agent_goal",
)

if st.button(
    "Run LangGraph Agent",
    type="primary",
    key="run_transaction_agent",
):
    if not account_id_agent.strip():
        st.warning("Please enter an account ID.")
    elif not agent_goal.strip():
        st.warning("Please enter an agent goal.")
    elif not mcp_tools:
        st.warning("Click 'Discover MCP Tools' first, and ensure the MCP server is running.")
    else:
        initial_graph_state: GraphAgentState = {
            "goal": agent_goal.strip(),
            "account_id": account_id_agent.strip(),
            "spending_limit": Decimal(str(spending_limit_input)),
            "transactions": [],
            "analysis": {},
            "balance_result": "",
            "observations": [],
            "actions": [],
            "next_action": "",
            "decision_reason": "",
            "step_count": 0,
            "status": "running",
            "errors": [],
            "final_answer": "",
        }

        try:
            with st.spinner("The LangGraph agent is deciding, acting, and reassessing..."):
                result = agent_graph.invoke(
                    initial_graph_state,
                    config={"recursion_limit": 30, "callbacks": [langfuse_handler]},
                )

            # Adapt LangGraph's dictionary state to the existing display dataclass.
            display_state = AgentState(goal=result.get("goal", agent_goal.strip()))
            display_state.plan = [
                "LLM decides which action is relevant to the goal",
                "Execute the selected MCP tool or Python analysis",
                "Record the observation and reassess the next action",
                "Stop when the goal is answered or a safety limit is reached",
            ]
            display_state.observations = result.get("observations", [])
            display_state.transactions = result.get("transactions", [])
            display_state.analysis = result.get("analysis", {})
            display_state.actions = result.get("actions", [])
            display_state.missing_information = result.get("errors", [])
            display_state.status = result.get("status", "needs_attention")
            display_state.final_answer = result.get("final_answer", "")

            successful_actions = _successful_action_names(result)
            if "get_account_balance" in successful_actions:
                display_state.completed_steps.append("Retrieve account balance through MCP")
            if "get_recent_transactions" in successful_actions:
                display_state.completed_steps.append("Retrieve recent transactions through MCP")
                display_state.completed_steps.append("Validate the retrieved transaction data")
            if "analyze_transactions" in successful_actions:
                display_state.completed_steps.extend([
                    "Calculate total expenses using Python",
                    "Identify the largest expense",
                    "Compare expenses against the spending limit",
                ])
            if display_state.final_answer:
                display_state.completed_steps.append("Prepare the final response")

            display_state.observations.append({
                "source": "agent_decision_summary",
                "step_count": result.get("step_count", 0),
                "last_decision_reason": result.get("decision_reason", ""),
            })
            st.session_state["agent_state"] = display_state
            st.session_state["agent_graph_state"] = result

        except Exception as exc:
            st.error(f"LangGraph agent execution failed: {exc}")


# ---------------------------------------------------------
# LangGraph Agent Results
# ---------------------------------------------------------

agent_state = st.session_state.get("agent_state")
agent_graph_state = st.session_state.get("agent_graph_state", {})

if agent_state:
    st.subheader("Agent Status")
    st.write(f"**Status:** {agent_state.status}")
    st.write(f"**Actions executed:** {agent_graph_state.get('step_count', 0)} / {MAX_AGENT_STEPS}")

    st.subheader("Workflow overview")
    for index, step in enumerate(agent_state.plan, start=1):
        st.write(f"{index}. {step}")

    if agent_graph_state.get("balance_result"):
        st.subheader("Account balance result")
        st.write(agent_graph_state["balance_result"])

    if agent_state.analysis:
        analysis = agent_state.analysis
        col1, col2, col3 = st.columns(3)
        col1.metric("Total expenses", f"£{analysis['total_expenses']:,.2f}")
        col2.metric("Largest expense", f"£{analysis['largest_expense']['amount']:,.2f}")
        col3.metric("Spending limit", f"£{analysis['spending_limit']:,.2f}")

        if analysis["exceeded_limit"]:
            st.warning(
                "The returned transactions exceed the spending limit by "
                f"£{analysis['amount_over_limit']:,.2f}."
            )
        else:
            remaining = analysis["spending_limit"] - analysis["total_expenses"]
            st.success(f"The returned transactions are £{remaining:,.2f} below the spending limit.")

        largest = analysis["largest_expense"]
        st.write(
            f"**Largest expense:** {largest['description']} on {largest['date']}"
        )

    if agent_state.missing_information:
        st.subheader("Issues / limitations")
        for item in agent_state.missing_information:
            st.warning(item)

    if agent_state.final_answer:
        st.subheader("Final Answer")
        st.markdown(agent_state.final_answer)

    with st.expander("LLM decisions and action history", expanded=True):
        st.json(agent_state.actions)

    with st.expander("Observations received by the agent"):
        st.json(agent_state.observations)

    with st.expander("Retrieved transactions"):
        for transaction in agent_state.transactions:
            st.write({
                **transaction,
                "amount": str(transaction["amount"]),
            })

    with st.expander("Full LangGraph state summary"):
        st.json({
            "goal": agent_graph_state.get("goal"),
            "next_action": agent_graph_state.get("next_action"),
            "decision_reason": agent_graph_state.get("decision_reason"),
            "step_count": agent_graph_state.get("step_count"),
            "status": agent_graph_state.get("status"),
            "errors": agent_graph_state.get("errors", []),
            "analysis": {
                key: str(value) if isinstance(value, Decimal) else value
                for key, value in agent_graph_state.get("analysis", {}).items()
            },
        })



# ---------------------------------------------------------
# 4. Multi-Agent Workflow — Supervisor + Specialist Agents
# ---------------------------------------------------------

st.divider()
st.header("4️⃣ Multi-Agent Workflow — LangGraph")
st.caption(
    "Part 1 learning focus: routing and collaboration. Watch which specialists are eligible, "
    "which one the supervisor selects, and how the result returns to the supervisor. "
    "Python remains responsible for exact financial calculations."
)

with st.expander("How this multi-agent workflow differs from the single-agent version"):
    st.markdown(
        "- **Supervisor Agent:** chooses which specialist should work next.\n"
        "- **Balance Agent:** retrieves the account balance through MCP and summarizes it.\n"
        "- **Transaction Agent:** retrieves and validates transactions through MCP.\n"
        "- **Budget Analysis Agent:** uses Python for exact calculations, then explains the results.\n"
        "- **Response Agent:** combines the specialist findings into a final answer.\n\n"
        "The agents use the same configured Groq model with different roles and prompts. "
        "They are coordinated by a single LangGraph StateGraph."
    )
    st.code(
        "START → Supervisor → [Balance Agent / Transaction Agent / Budget Agent] → Supervisor\n"
        "Supervisor → Response Agent → END",
        language="text",
    )

multi_account_id = st.text_input(
    "Multi-agent account ID",
    value="12345",
    key="multiagent_account_id",
)
multi_spending_limit = st.number_input(
    "Multi-agent spending limit (£)",
    min_value=0.0,
    value=200.0,
    step=10.0,
    key="multiagent_spending_limit",
)
multi_agent_goal = st.text_area(
    "Goal for the multi-agent team",
    value=(
        "Check my current account balance, analyze my recent transactions, "
        "identify my largest expense, and check whether I exceeded my £200 spending limit."
    ),
    key="multiagent_goal",
)

if st.button("Run Multi-Agent Workflow", type="primary", key="run_multiagent_workflow"):
    if not multi_account_id.strip():
        st.warning("Please enter an account ID.")
    elif not multi_agent_goal.strip():
        st.warning("Please enter a goal for the multi-agent team.")
    elif not mcp_tools:
        st.warning("Click 'Discover MCP Tools' in section 1 and ensure the MCP server is running.")
    else:
        initial_multi_state: MultiAgentState = {
            "goal": multi_agent_goal.strip(),
            "account_id": multi_account_id.strip(),
            "spending_limit": Decimal(str(multi_spending_limit)),
            "balance_result": "",
            "transactions": [],
            "analysis": {},
            "specialist_reports": {},
            "completed_agents": [],
            "next_agent": "",
            "supervisor_reason": "",
            "agent_trace": [],
            "step_count": 0,
            "errors": [],
            "final_answer": "",
            "status": "running",
        }
        try:
            with st.spinner("Supervisor is coordinating the specialist agents..."):
                multi_result = multiagent_graph.invoke(
                    initial_multi_state,
                    config={"recursion_limit": 40, "callbacks": [langfuse_handler]},
                )
            st.session_state["multiagent_result"] = multi_result
        except Exception as exc:
            st.error(f"Multi-agent workflow failed: {exc}")

multi_result = st.session_state.get("multiagent_result")
if multi_result:
    st.subheader("Multi-Agent Run Summary")
    col1, col2, col3 = st.columns(3)
    col1.metric("Status", multi_result.get("status", "unknown"))
    col2.metric("Specialist steps", multi_result.get("step_count", 0))
    col3.metric("Agents completed", len(multi_result.get("completed_agents", [])))

    st.write("**Specialists invoked:**", ", ".join(multi_result.get("completed_agents", [])) or "None")
    if multi_result.get("balance_result"):
        st.write("**Account balance result:**", multi_result["balance_result"])
    if multi_result.get("analysis"):
        analysis = multi_result["analysis"]
        col1, col2, col3 = st.columns(3)
        col1.metric("Total expenses", f"£{analysis['total_expenses']:,.2f}")
        col2.metric("Largest expense", f"£{analysis['largest_expense']['amount']:,.2f}")
        col3.metric("Spending limit", f"£{analysis['spending_limit']:,.2f}")

    st.subheader("Final Answer")
    st.markdown(multi_result.get("final_answer", "No final answer was generated."))

    with st.expander("🧭 Routing and Collaboration Timeline", expanded=True):
        st.caption(
            "Follow the supervisor's decisions and each specialist's outcome. "
            "Available agents are candidates; the decision column shows the route selected."
        )
        timeline_rows = []
        for event_number, event in enumerate(multi_result.get("agent_trace", []), start=1):
            role = event.get("role", "unknown")
            if role == "supervisor":
                timeline_rows.append({
                    "Step": event_number,
                    "Event": "Supervisor decision",
                    "Selected route": event.get("decision", "not recorded"),
                    "Outcome / reason": event.get("reason", ""),
                    "Available agents": ", ".join(event.get("available_agents", [])),
                })
            else:
                timeline_rows.append({
                    "Step": event_number,
                    "Event": role,
                    "Selected route": "—",
                    "Outcome / reason": event.get("finding", event.get("summary", event.get("error", event.get("status", "")))),
                    "Available agents": "—",
                })
        if timeline_rows:
            st.dataframe(timeline_rows, use_container_width=True, hide_index=True)
        else:
            st.info("No routing events were recorded for this run.")

    with st.expander("Raw supervisor decisions and agent trace"):
        st.json(multi_result.get("agent_trace", []))
    with st.expander("Specialist reports"):
        st.json(multi_result.get("specialist_reports", {}))
    with st.expander("Shared multi-agent state"):
        safe_analysis = {
            key: str(value) if isinstance(value, Decimal) else value
            for key, value in multi_result.get("analysis", {}).items()
        }
        st.json({
            "goal": multi_result.get("goal"),
            "completed_agents": multi_result.get("completed_agents", []),
            "next_agent": multi_result.get("next_agent"),
            "step_count": multi_result.get("step_count", 0),
            "status": multi_result.get("status"),
            "errors": multi_result.get("errors", []),
            "analysis": safe_analysis,
        })
    if multi_result.get("errors"):
        st.warning("Some specialist steps reported issues. Review the trace and reports above.")


# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------

st.divider()

st.caption(
   
    "AI Agents."
)

