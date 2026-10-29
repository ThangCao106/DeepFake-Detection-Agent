"""
Gemini SDK & Google GenAI Backend for Observability Agent (Tier 1 Engine).

Implements Option C Tiered Hybrid Architecture:
  - Lead Forensic Examiner & Fast Telemetry Engine: Google Gemini 2.5 Flash / Pro, Gemini 2.0 Flash
  - Streaming support for OpenAI-compatible completions.
  - Automatic fallback across model generations (2.5 -> 2.0 -> 1.5).
"""
import os
import json
import asyncio
import urllib.request
import urllib.error
from typing import AsyncIterator, Optional, Dict, Any, List

_CACHED_ACTIVE_MODELS: list[str] = []

DEFAULT_GEMINI_MODEL = "gemini-3.8-flash"

SUPPORTED_GEMINI_MODELS = [
    "gemini-3.8-flash",
    "gemini-3.8-flash-cyber",
    "gemini-2.5-flash",
    "gemini-2.5-pro",
    "gemini-2.0-flash",
    "gemini-1.5-flash"
]

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

    if key.startswith("AQ.") or key.startswith("AQ"):
        return {
            "valid": False,
            "status": "INVALID_KEY",
            "message": "Mã token dạng AQ.Ab8... là OAuth refresh token, không thể dùng trực tiếp cho Google GenAI API. Vui lòng tạo API Key chuẩn (bắt đầu bằng AIzaSy...) tại https://aistudio.google.com/apikey",
            "masked": masked
        }

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
                "User-Agent": "Deepfake-Forensic-Agent/3.0",
                "x-goog-api-key": key
            }
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=8) as resp:
                if resp.status == 200:
                    try:
                        res_json = json.loads(resp.read().decode("utf-8"))
                        live_models = [
                            m.get("name", "").replace("models/", "")
                            for m in res_json.get("models", [])
                            if "generateContent" in m.get("supportedGenerationMethods", [])
                        ]
                        global _CACHED_ACTIVE_MODELS
                        if live_models:
                            _CACHED_ACTIVE_MODELS = live_models
                    except Exception:
                        pass
                    return {
                        "valid": True,
                        "status": "AUTHENTICATED",
                        "message": "API Key hợp lệ và hoạt động bình thường",
                        "masked": masked,
                        "active_models_count": len(_CACHED_ACTIVE_MODELS)
                    }
        except urllib.error.HTTPError as e:
            if e.code in [400, 401, 403]:
                return {
                    "valid": False,
                    "status": "INVALID_KEY",
                    "message": f"API Key không hợp lệ hoặc bị từ chối (HTTP {e.code}). Vui lòng tạo API key chuẩn (bắt đầu bằng AIzaSy...) tại https://aistudio.google.com/apikey",
                    "masked": masked
                }
            return {
                "valid": False,
                "status": "ERROR",
                "message": f"HTTP {e.code} lỗi xác thực API",
                "masked": masked
            }
        except Exception as ex:
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
    """Execute one-shot generation using Google Gemini API with smart fallback."""
    key = api_key or get_gemini_api_key()
    if not key:
        raise ValueError("GEMINI_API_KEY chưa được cấu hình. Vui lòng cập nhật API Key tại menu 'Cài đặt kết nối'.")

    if key.startswith("AQ.") or key.startswith("AQ"):
        raise ValueError(
            "GEMINI_API_KEY hiện tại là token dạng OAuth ('AQ.Ab8...'), không thể dùng cho Google GenAI API. "
            "Vui lòng vào https://aistudio.google.com/apikey tạo API Key chuẩn (bắt đầu bằng AIzaSy...) và lưu vào Cài đặt."
        )

    # Normalize requested model
    model = model.strip() if model else DEFAULT_GEMINI_MODEL
    # Aliases
    if model in ["gemini-flash", "fast", "l1"]:
        model = "gemini-2.5-flash"
    elif model in ["gemini-pro", "deep"]:
        model = "gemini-2.5-pro"

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
        "x-goog-api-key": key,
        "User-Agent": "Deepfake-Forensic-Agent/3.0"
    }

    loop = asyncio.get_running_loop()

    def _sync_post():
        primary_url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}"
        primary_req = urllib.request.Request(primary_url, data=data_bytes, headers=req_headers, method="POST")

        try:
            with urllib.request.urlopen(primary_req, timeout=30) as resp:
                res_body = resp.read().decode("utf-8")
                return json.loads(res_body)
        except urllib.error.HTTPError as e:
            err_text = e.read().decode("utf-8", errors="ignore")
            if e.code == 401:
                raise RuntimeError(
                    f"Gemini API Error (HTTP 401): API key không hợp lệ hoặc không có quyền truy cập. "
                    f"Vui lòng tạo Google AI Studio API Key (bắt đầu bằng AIzaSy...) tại https://aistudio.google.com/apikey"
                )

            # Auto-fallback on 404 (model not in beta yet), 503 (high demand), 429 (rate-limited), or 500
            if e.code in [404, 429, 500, 502, 503, 504]:
                fallback_candidates = [
                    "gemini-3.8-flash", "gemini-2.5-flash", "gemini-2.0-flash",
                    "gemini-1.5-flash", "gemini-flash-latest"
                ]
                for fb_model in fallback_candidates:
                    if fb_model == model:
                        continue
                    try:
                        fb_url = f"https://generativelanguage.googleapis.com/v1beta/models/{fb_model}:generateContent?key={key}"
                        fb_req = urllib.request.Request(fb_url, data=data_bytes, headers=req_headers, method="POST")
                        with urllib.request.urlopen(fb_req, timeout=20) as fb_resp:
                            fb_body = fb_resp.read().decode("utf-8")
                            parsed = json.loads(fb_body)
                            if parsed.get("candidates"):
                                return parsed
                    except Exception:
                        continue

            if e.code == 503:
                raise RuntimeError(
                    "Google Gemini đang gặp tình trạng quá tải tạm thời (HTTP 503: High demand). "
                    "Hệ thống đã thử các model dự phòng nhưng cụm máy chủ Google vẫn đang bận. "
                    "Vui lòng thử lại sau 5–10 giây hoặc chọn model khác."
                )
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
        return "⚠️ Gemini không phản hồi nội dung văn bản."
    except Exception as e_parse:
        return f"⚠️ Lỗi phân tích phản hồi từ Gemini: {e_parse}\nRaw: {res_json}"


async def stream_gemini_generate(
    prompt: str,
    system_instruction: Optional[str] = None,
    model: str = DEFAULT_GEMINI_MODEL,
    api_key: Optional[str] = None,
    temperature: float = 0.2
) -> AsyncIterator[str]:
    """Execute streaming generation via word slicing over one-shot generation."""
    full_text = await run_gemini_generate(
        prompt=prompt,
        system_instruction=system_instruction,
        model=model,
        api_key=api_key,
        temperature=temperature
    )

    words = full_text.split(" ")
    for i, word in enumerate(words):
        chunk = word + (" " if i < len(words) - 1 else "")
        yield chunk
        await asyncio.sleep(0.015)
