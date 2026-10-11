"""Tiny MCP client used for the teaching demo.

Run the MCP server first:
    python server.py

Then:
    python client.py
"""

import asyncio
from mcp import Client

MCP_SERVER_URL = "http://127.0.0.1:8000/mcp"

async def main() -> None:
    async with Client(MCP_SERVER_URL) as client:
        print("Connected to:", client.server_info)
        print("Protocol:", client.protocol_version)
        print("\nDiscovered tools:")

        tools = await client.list_tools()
        for tool in tools.tools:
            print(f"- {tool.name}: {tool.description}")

        print("\nCalling get_account_balance...")
        result = await client.call_tool(
            "get_account_balance",
            {"account_id": "12345"},
        )
        print(result)


if __name__ == "__main__":
    asyncio.run(main())
