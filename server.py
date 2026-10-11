"""Banking MCP Server - live demo.

Run:
    python server.py

The server exposes the banking capabilities through MCP over Streamable HTTP.
"""

from mcp.server.mcpserver import MCPServer

mcp = MCPServer(
    "Banking Demo Server",
    instructions="Demo banking capabilities for the MCP teaching session.",
)


@mcp.tool()
def get_account_balance(account_id: str) -> str:
    """Get the current account balance for a customer."""
    mock_accounts = {
        "12345": 85420.50,
        "67890": 12450.75,
    }

    balance = mock_accounts.get(account_id)

    if balance is None:
        return f"Account {account_id} was not found."

    return f"Account {account_id} has a current balance of £{balance:,.2f}."


@mcp.tool()
def get_recent_transactions(
    account_id: str,
    number_of_transactions: int = 5,
) -> str:
    """Get recent transactions for a customer account."""
    mock_transactions = {
        "12345": [
            {"date": "2026-10-04", "description": "Tesco", "amount": "-45.20"},
            {"date": "2026-10-03", "description": "Salary Credit", "amount": "3200.00"},
            {"date": "2026-10-02", "description": "Amazon", "amount": "-82.50"},
            {"date": "2026-10-01", "description": "Electricity Bill", "amount": "-120.00"},
            {"date": "2026-09-30", "description": "Coffee Shop", "amount": "-5.80"},
        ]
    }

    transactions = mock_transactions.get(account_id)

    if transactions is None:
        return f"No transaction data found for account {account_id}."

    number_of_transactions = max(1, min(number_of_transactions, 10))
    selected = transactions[:number_of_transactions]

    lines = [
        f"{item['date']} | {item['description']} | £{item['amount']}"
        for item in selected
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    print("Starting Banking MCP Server on http://127.0.0.1:8000/mcp")
    mcp.run(
        transport="streamable-http",
        host="127.0.0.1",
        port=8000,
    )
