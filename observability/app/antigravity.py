"""Adapter for Antigravity SDK integration.

This module provides a minimal async-compatible wrapper so the rest of the
agent can call Antigravity as an alternate model backend. It is intentionally
lightweight: if the real SDK is installed, it will use it; otherwise it
exposes a stub that raises ImportError when invoked.

Design notes:
- Exposes `run_antigravity(prompt, **opts) -> str` for one-shot prompts.
- Exposes `AntigravityClient` with `connect()`/`disconnect()`/`query()` to
  mirror the `ClaudeSDKClient` usage pattern in `app/agent.py`.
"""
from typing import Any, Optional
import asyncio

try:
    # Attempt to import the real SDK (package name from upstream repo may vary).
    # The git dependency in pyproject.toml uses `antigravity-sdk` as the wheel name;
    # if upstream package uses a different import path, adjust accordingly.
    import antigravity_sdk as ag
except Exception:  # pragma: no cover - runtime import guard
    ag = None


class AntigravityClient:
    """Minimal client wrapper matching the connect/query/disconnect pattern.

    If the real SDK is not installed, constructing or calling methods raises
    ImportError so callers know to install the optional dependency.
    """

    def __init__(self, *, api_key: Optional[str] = None, **kwargs: Any):
        if ag is None:
            raise ImportError("antigravity-sdk is not installed. Install extras 'antigravity'.")
        # Store config for connection; real SDK may accept api_key or different options.
        self._client = ag.Client(api_key=api_key) if hasattr(ag, "Client") else None

    async def connect(self) -> None:
        # If the SDK requires async connect, call it; otherwise no-op.
        if hasattr(self._client, "connect"):
            maybe = self._client.connect()
            if asyncio.iscoroutine(maybe):
                await maybe

    async def disconnect(self) -> None:
        if hasattr(self._client, "disconnect"):
            maybe = self._client.disconnect()
            if asyncio.iscoroutine(maybe):
                await maybe

    async def query(self, prompt: str, **opts: Any) -> str:
        """Send a single prompt and return text reply.

        The real SDK may provide streaming; this adapter returns the final
        text result to match run_agent()/stream_turn() usage patterns.
        """
        if self._client is None:
            raise ImportError("antigravity-sdk client not available")
        # Prefer sync `generate` or async `agenerate` depending on SDK.
        if hasattr(self._client, "generate"):
            result = self._client.generate(prompt=prompt, **opts)
            if asyncio.iscoroutine(result):
                result = await result
            # Assume result has a `.text` attribute or is a string.
            return getattr(result, "text", str(result))
        if hasattr(self._client, "agenerate"):
            res = await self._client.agenerate(prompt=prompt, **opts)
            return getattr(res, "text", str(res))
        # Fallback: try a top-level wrapper function
        if hasattr(ag, "run"):
            maybe = ag.run(prompt=prompt, **opts)
            if asyncio.iscoroutine(maybe):
                maybe = await maybe
            return str(maybe)
        raise RuntimeError("antigravity-sdk does not expose a supported API shape")


async def run_antigravity(prompt: str, *, api_key: Optional[str] = None, **opts: Any) -> str:
    """Convenience one-shot call used by /api/alert or similar flows.

    This mirrors `run_agent` in `app/agent.py` and returns final text.
    """
    if ag is None:
        raise ImportError("antigravity-sdk is not installed. Add the antigravity extra and reinstall.")
    client = AntigravityClient(api_key=api_key)
    await client.connect()
    try:
        out = await client.query(prompt, **opts)
    finally:
        await client.disconnect()
    return out
