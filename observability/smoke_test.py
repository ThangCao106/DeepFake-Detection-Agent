"""
Smoke Test for Observability SRE Agent (Dual SDK: Claude & Gemini).
Verifies connectivity, authentication, and core SRE capabilities without external DB dependencies.
"""
import os
import sys
import asyncio
from pathlib import Path
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(PROJECT_ROOT / ".env")

async def test_gemini() -> bool:
    print("\n🔍 [1/2] Testing Google Gemini SDK (Tier 1 Engine)...")
    gemini_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not gemini_key:
        print("  ⚠️ GEMINI_API_KEY not found in environment or .env. Skipping live call.")
        return False

    masked = gemini_key[:8] + "..." if len(gemini_key) > 8 else "***"
    print(f"  🔑 Found GEMINI_API_KEY: {masked}")

    try:
        from app.gemini_backend import run_gemini_generate
        reply = await run_gemini_generate(
            prompt="SRE ping test: return 'PONG_SRE_OK' and one status sentence.",
            model="gemini-2.5-flash"
        )
        print(f"  ✅ Gemini Response: {reply.strip()[:100]}...")
        return True
    except Exception as e:
        print(f"  ❌ Gemini Test Notice: {e}")
        return False

async def test_claude() -> bool:
    print("\n🔍 [2/2] Testing Claude Agent SDK (Tier 2 Engine)...")
    try:
        from claude_agent_sdk import ClaudeAgentOptions, query, AssistantMessage, TextBlock
        from app.config import ALLOWED_TOOLS, DISALLOWED_TOOLS

        options = ClaudeAgentOptions(
            cwd=str(PROJECT_ROOT),
            system_prompt="You are an SRE assistant. Reply concisely.",
            allowed_tools=ALLOWED_TOOLS,
            disallowed_tools=DISALLOWED_TOOLS,
            permission_mode="default",
            include_partial_messages=False,
        )

        print(f"  🛠️ SRE Whitelisted Tools: {ALLOWED_TOOLS}")
        async for message in query(prompt="SRE ping: reply with 'CLAUDE_SRE_OK'.", options=options):
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        print(f"  ✅ Claude Response: {block.text.strip()}")
                        return True
        return True
    except Exception as e:
        err_msg = str(e)
        if "not logged in" in err_msg.lower() or "clinotfounderror" in err_msg.lower():
            print("  ⚠️ Claude Code CLI not logged in on this machine. Run 'claude login' or set CLAUDE_API_KEY in .env.")
        else:
            print(f"  ⚠️ Claude Test Notice: {err_msg}")
        return False

async def main():
    print("=" * 70)
    print("🚀 SRE OBSERVABILITY AGENT — DUAL-ENGINE SMOKE TEST")
    print(f"   Working Directory: {PROJECT_ROOT}")
    print("=" * 70)

    gemini_ok = await test_gemini()
    claude_ok = await test_claude()

    print("\n" + "=" * 70)
    print("📊 SMOKE TEST SUMMARY:")
    print(f"   - Google Gemini Engine (Tier 1): {'🟢 ACTIVE' if gemini_ok else '⚪ NOT CONFIGURED'}")
    print(f"   - Claude Agent SDK Engine (Tier 2): {'🟢 ACTIVE' if claude_ok else '⚪ NEED LOGIN / API KEY'}")
    print("=" * 70)

if __name__ == "__main__":
    asyncio.run(main())
