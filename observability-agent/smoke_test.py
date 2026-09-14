import asyncio
from pathlib import Path

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ResultMessage,
    StreamEvent,
    SystemMessage,
    TextBlock,
    query,
)

PROJECT_ROOT = str(Path(__file__).resolve().parent)

CLICKHOUSE_MCP = {
    "clickhouse": {
        "command": "uvx",
        "args": ["--with", "mcp-clickhouse", "--python", "3.10", "mcp-clickhouse"],
        "env": {
            "CLICKHOUSE_HOST": "192.168.53.251",
            "CLICKHOUSE_PORT": "8123",
            "CLICKHOUSE_USER": "default",
            "CLICKHOUSE_PASSWORD": "",
            "CLICKHOUSE_SECURE": "false",
        },
    }
}


async def main() -> None:
    options = ClaudeAgentOptions(
        cwd=PROJECT_ROOT,
        system_prompt="You are an SRE assistant.",
        mcp_servers=CLICKHOUSE_MCP,
        allowed_tools=["mcp__clickhouse__*", "mcp__claude-mem__*"],
        permission_mode="default",
        include_partial_messages=True,
        # Needed to pick up claude-mem's hooks/MCP registration from $HOME's Claude
        # Code settings (see README "claude-mem" section) — without this, `mcp_servers`
        # below only shows "clickhouse", never claude-mem's, even if it's installed.
        setting_sources=["user"],
    )
    async for message in query(prompt="Say hi in one short sentence.", options=options):
        if isinstance(message, StreamEvent):
            print("STREAM_EVENT:", message.event)
        elif isinstance(message, SystemMessage):
            if message.subtype == "init":
                # Confirms exactly which MCP servers/tools are actually live — use this
                # to verify claude-mem's real server name after `npx claude-mem install`.
                print("MCP servers:", message.data.get("mcp_servers"))
                print("Tools:", message.data.get("tools"))
        elif isinstance(message, AssistantMessage):
            for block in message.content:
                if isinstance(block, TextBlock):
                    print("FINAL TEXT BLOCK:", block.text)
        elif isinstance(message, ResultMessage):
            print("Done:", message.subtype)


if __name__ == "__main__":
    asyncio.run(main())
