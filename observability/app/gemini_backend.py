"""
Gemini SDK & Google GenAI Backend for Observability Agent (Tier 1 Engine).

Implements Option C Tiered Hybrid Architecture:
  - Tier 1 (Gemini 2.5 Flash / Pro): Fast telemetry triaging, real-time log scanning,
    and high-throughput query synthesis using Google GenAI API.
  - Streaming support for OpenAI-compatible completions.
  - Automatic fallback & ClickHouse query fast-path.
"""
import os
import json
import asyncio
import urllib.request
import urllib.error
from typing import AsyncIterator, Optional, Dict, Any, List

DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"

def get_gemini_api_key() -> str:
    """Retrieve Gemini API key from environment."""
    return os.environ.get("GEMINI_API_KEY", "").strip()

def is_gemini_configured() -> bool:
    """Check if a valid Gemini API key is configured."""
    return bool(get_gemini_api_key())

async def run_gemini_generate(
    prompt: str,
    system_instruction: Optional[str] = None,
    model: str = DEFAULT_GEMINI_MODEL,
    api_key: Optional[str] = None,
    temperature: float = 0.2
) -> str:
    """Execute one-shot generation using Google Gemini API."""
    key = api_key or get_gemini_api_key()
    if not key:
        raise ValueError("GEMINI_API_KEY is not configured. Please set it in .env or via Web UI.")

    # Normalize model ID
    if model in ["gemini-flash", "fast", "l1"]:
        model = "gemini-2.5-flash"
    elif model in ["gemini-pro", "deep"]:
        model = "gemini-2.5-pro"

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}"

    payload: Dict[str, Any] = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": prompt}]
            }
        ],
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": 4096
        }
    }

    if system_instruction:
        payload["systemInstruction"] = {
            "parts": [{"text": system_instruction}]
        }

    data_bytes = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data_bytes,
        headers={"Content-Type": "application/json"},
        method="POST"
    )

    loop = asyncio.get_running_loop()

    def _sync_post():
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                res_body = resp.read().decode("utf-8")
                return json.loads(res_body)
        except urllib.error.HTTPError as e:
            err_text = e.read().decode("utf-8", errors="ignore")
            # If 2.5-flash is not yet activated on the API key tier, fallback to 1.5-flash
            if e.code == 404 and "2.5" in model:
                fallback_model = "gemini-1.5-flash"
                fb_url = f"https://generativelanguage.googleapis.com/v1beta/models/{fallback_model}:generateContent?key={key}"
                fb_req = urllib.request.Request(fb_url, data=data_bytes, headers={"Content-Type": "application/json"}, method="POST")
                with urllib.request.urlopen(fb_req, timeout=30) as fb_resp:
                    return json.loads(fb_resp.read().decode("utf-8"))
            raise RuntimeError(f"Gemini API Error (HTTP {e.code}): {err_text}")
        except Exception as ex:
            raise RuntimeError(f"Gemini Request Failed: {ex}")

    res_json = await loop.run_in_executor(None, _sync_post)

    try:
        candidates = res_json.get("candidates", [])
        if candidates:
            parts = candidates[0].get("content", {}).get("parts", [])
            if parts:
                return "".join(p.get("text", "") for p in parts)
        return "⚠️ Gemini returned an empty response."
    except Exception as e_parse:
        return f"⚠️ Error parsing Gemini response: {e_parse}\nRaw: {res_json}"

async def stream_gemini_generate(
    prompt: str,
    system_instruction: Optional[str] = None,
    model: str = DEFAULT_GEMINI_MODEL,
    api_key: Optional[str] = None,
    temperature: float = 0.2
) -> AsyncIterator[str]:
    """Stream text tokens from Gemini."""
    # To ensure stable streaming over standard HTTP without server-sent-event chunking issues on all OSs,
    # we yield initial triaging header then stream the generated tokens smoothly.
    yield "⚡ **[Gemini Tier 1 Engine]** Telemetry Triaging...\n\n"
    
    full_text = await run_gemini_generate(
        prompt=prompt,
        system_instruction=system_instruction,
        model=model,
        api_key=api_key,
        temperature=temperature
    )
    
    # Stream in word/line chunks
    words = full_text.split(" ")
    buffer = []
    for w in words:
        buffer.append(w)
        if len(buffer) >= 4:
            yield " ".join(buffer) + " "
            buffer = []
            await asyncio.sleep(0.015)
    if buffer:
        yield " ".join(buffer)
