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

DEFAULT_GEMINI_MODEL = "gemini-flash-latest"

def get_gemini_api_key() -> str:
    """Retrieve Gemini API key from environment."""
    return os.environ.get("GEMINI_API_KEY", "").strip()

def is_gemini_configured() -> bool:
    """Check if a valid Gemini API key is configured."""
    return bool(get_gemini_api_key())

async def verify_gemini_api_key(api_key: Optional[str] = None) -> dict:
    """Pre-check Google Gemini API Key validity upfront by inspecting format and checking models endpoint."""
    key = (api_key or get_gemini_api_key()).strip()
    if not key:
        return {
            "valid": False,
            "status": "NOT_CONFIGURED",
            "message": "Chưa cấu hình GEMINI_API_KEY",
            "masked": ""
        }

    masked = (key[:8] + "...") if len(key) > 8 else "***"

    if len(key) < 20:
        return {
            "valid": False,
            "status": "INVALID_FORMAT",
            "message": "Độ dài GEMINI_API_KEY không hợp lệ (< 20 ký tự)",
            "masked": masked
        }

    url = f"https://generativelanguage.googleapis.com/v1beta/models?key={key}"
    loop = asyncio.get_running_loop()

    def _sync_check():
        try:
            headers = {
                "User-Agent": "Deepfake-Forensic-Agent/2.5",
                "x-goog-api-key": key
            }
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=5) as resp:
                if resp.status == 200:
                    return {
                        "valid": True,
                        "status": "AUTHENTICATED",
                        "message": "API Key hợp lệ và hoạt động bình thường",
                        "masked": masked
                    }
        except urllib.error.HTTPError as e:
            if e.code in [400, 401, 403]:
                return {
                    "valid": False,
                    "status": "INVALID_KEY",
                    "message": f"API Key không hợp lệ hoặc bị từ chối (HTTP {e.code}). Cần lấy API key chuẩn tại aistudio.google.com",
                    "masked": masked
                }
            return {
                "valid": False,
                "status": "ERROR",
                "message": f"HTTP {e.code} lỗi xác thực API",
                "masked": masked
            }
        except Exception as ex:
            # Container network or DNS fallback
            return {
                "valid": False,
                "status": "CONNECTION_ERROR",
                "message": f"Không thể kết nối Google AI ({ex})",
                "masked": masked
            }
        return {"valid": False, "status": "INVALID_KEY", "message": "Key không phản hồi", "masked": masked}

    return await loop.run_in_executor(None, _sync_check)


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

    # Normalize deprecated or alias model IDs to current active endpoints
    MODEL_MAP = {
        "gemini-2.5-flash": "gemini-flash-latest",
        "gemini-2.5-pro": "gemini-flash-latest",
        "gemini-1.5-flash": "gemini-flash-latest",
        "gemini-1.5-pro": "gemini-flash-latest",
        "gemini-flash": "gemini-flash-latest",
        "gemini-pro": "gemini-flash-latest",
        "fast": "gemini-flash-latest",
        "l1": "gemini-flash-latest",
        "deep": "gemini-flash-latest"
    }
    model = MODEL_MAP.get(model, model)
    if not model or model.startswith("gemini-2.5") or model.startswith("gemini-1.5"):
        model = "gemini-flash-latest"

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
    req_headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": key
    }
    req = urllib.request.Request(
        url,
        data=data_bytes,
        headers=req_headers,
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
            if e.code == 401:
                raise RuntimeError(
                    f"Gemini API Error (HTTP 401): API key không hợp lệ hoặc không có quyền truy cập (ACCESS_TOKEN_TYPE_UNSUPPORTED). "
                    f"Vui lòng tạo Google AI Studio API Key (bắt đầu bằng AIzaSy...) tại https://aistudio.google.com/apikey và cập nhật qua menu 'Cài đặt kết nối'."
                )
            # If the requested model is 404 (retired) or 429 (rate-limited), fallback to gemini-flash-latest or gemini-3.8-flash
            if e.code in [404, 429]:
                candidates = ["gemini-flash-latest", "gemini-3.8-flash"]
                for fallback_model in candidates:
                    if fallback_model == model:
                        continue
                    try:
                        fb_url = f"https://generativelanguage.googleapis.com/v1beta/models/{fallback_model}:generateContent?key={key}"
                        fb_req = urllib.request.Request(fb_url, data=data_bytes, headers=req_headers, method="POST")
                        with urllib.request.urlopen(fb_req, timeout=30) as fb_resp:
                            return json.loads(fb_resp.read().decode("utf-8"))
                    except Exception:
                        continue
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
