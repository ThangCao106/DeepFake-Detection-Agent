"""OpenAI-compatible /v1/chat/completions + /v1/models so LibreChat's
'Custom Endpoint' can talk to this agent the same way it talks to Ollama."""

import hashlib
import json
import time
import uuid
from typing import Any

from fastapi import APIRouter, Depends, Header, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.agent import UserContent, drop_client, get_client, rekey_client, stream_turn
from app import antigravity, gemini_backend
from app.config import API_KEY, CHAT_SYSTEM_PROMPT

router = APIRouter()

MODEL_ID = "sre-agent-hybrid"
LEGACY_MODEL_ID = "observability-agent"
ANTIGRAVITY_MODEL_ID = "antigravity-agent"
SUPPORTED_MODELS = [
    "sre-agent-hybrid",
    "gemini-2.5-flash",
    "gemini-2.5-pro",
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
    "gemini-1.5-flash",
    "gemini-1.5-pro",
    "claude-3-7-sonnet",
    "claude-3-7-sonnet-20250219",
    "claude-3-5-sonnet",
    "meta-muse-spark",
    "meta-muse-wonder",
    "meta-muse-glimmer-30b",
    "observability-agent",
    "antigravity-agent"
]

# LibreChat resends the full message history every request (no server-side
# session concept on its side for custom endpoints). We fingerprint the prior
# messages to recognize "this is a continuation of a conversation we already
# have a connected client for" — see app/agent.py for why a connected client is
# reused instead of spawning a fresh subprocess per message.


def check_bearer(authorization: str = Header(default="")) -> None:
    token = authorization.removeprefix("Bearer ").strip()
    if token != API_KEY:
        raise HTTPException(status_code=401, detail="invalid api key")


class ChatMessage(BaseModel):
    role: str
    # LibreChat gửi content dạng string cho tin nhắn text thuần, nhưng dạng list các content
    # block (chuẩn multimodal OpenAI: {"type":"text",...} / {"type":"image_url",...}) khi có
    # đính kèm ảnh — trước đây field này chỉ nhận str nên request có ảnh bị Pydantic từ chối
    # thẳng với 422 (không tới được code bên dưới, không liên quan gì tới ALLOWED_TOOLS).
    content: str | list[dict[str, Any]]


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    stream: bool = False


def _content_to_text(content: str | list[dict[str, Any]]) -> str:
    """Chỉ lấy phần text — dùng cho fingerprint/transcript, KHÔNG cần ảnh ở đây."""
    if isinstance(content, str):
        return content
    return "\n".join(
        block.get("text", "") for block in content if isinstance(block, dict) and block.get("type") == "text"
    )


def _openai_content_to_user_content(content: str | list[dict[str, Any]]) -> UserContent:
    """Convert content chuẩn OpenAI (multimodal) sang UserContent chuẩn Anthropic mà
    agent.py/claude_agent_sdk hiểu — khác định dạng nhau (image_url + data: URI vs
    image + source.base64), cần map thủ công."""
    if isinstance(content, str):
        return content
    blocks: list[dict[str, Any]] = []
    for block in content:
        block_type = block.get("type")
        if block_type == "text":
            blocks.append({"type": "text", "text": block.get("text", "")})
        elif block_type == "image_url":
            url = block.get("image_url", {}).get("url", "")
            if url.startswith("data:"):
                header, _, data = url.partition(",")
                media_type = header.removeprefix("data:").split(";")[0] or "image/png"
                blocks.append(
                    {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": data}}
                )
            else:
                # URL thật (không phải data: base64) — agent không có tool để tự tải về
                # (không cấp WebFetch, xem DISALLOWED_TOOLS), nên chỉ note lại, không bịa
                # nội dung ảnh.
                blocks.append({"type": "text", "text": f"[ảnh qua URL, không tải được: {url}]"})
    return blocks


def _fingerprint(messages: list[ChatMessage]) -> str:
    payload = json.dumps(
        [{"role": m.role, "content": m.content} for m in messages],
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _transcript(messages: list[ChatMessage]) -> str:
    lines = []
    for m in messages:
        speaker = "User" if m.role == "user" else "Assistant"
        lines.append(f"{speaker}: {_content_to_text(m.content)}")
    return "\n".join(lines)


def _extract_system_prompt(messages: list[ChatMessage]) -> str:
    system_parts = [_content_to_text(m.content) for m in messages if m.role == "system"]
    if system_parts:
        return "\n".join(system_parts)
    return CHAT_SYSTEM_PROMPT


@router.get("/v1/models", dependencies=[Depends(check_bearer)])
async def list_models() -> dict:
    now = int(time.time())
    return {
        "object": "list",
        "data": [
            {
                "id": mid,
                "object": "model",
                "created": now,
                "owned_by": "sre-agent",
            }
            for mid in SUPPORTED_MODELS
        ],
    }


@router.post("/v1/chat/completions", dependencies=[Depends(check_bearer)])
async def chat_completions(req: ChatCompletionRequest):
    system_prompt = _extract_system_prompt(req.messages)
    history = [m for m in req.messages if m.role in ("user", "assistant")]
    completion_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())

    # --- TIER 1: GEMINI SDK ROUTING ---
    is_gemini_target = (
        req.model.lower().startswith("gemini")
        or req.model == ANTIGRAVITY_MODEL_ID
    )

    if is_gemini_target:
        prompt_text = _transcript(req.messages)
        gem_model = req.model if req.model.startswith("gemini") else "gemini-flash-latest"

        if not req.stream:
            try:
                reply_text = await gemini_backend.run_gemini_generate(
                    prompt=prompt_text,
                    system_instruction=system_prompt,
                    model=gem_model
                )
            except Exception as exc:
                raise HTTPException(status_code=502, detail=f"Gemini Engine error: {exc}") from exc

            return {
                "id": completion_id,
                "object": "chat.completion",
                "created": created,
                "model": req.model,
                "choices": [
                    {"index": 0, "message": {"role": "assistant", "content": reply_text}, "finish_reason": "stop"}
                ],
            }

        async def sse_gemini():
            first_chunk = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": req.model,
                "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
            }
            yield f"data: {json.dumps(first_chunk)}\n\n"

            try:
                async for text in gemini_backend.stream_gemini_generate(
                    prompt=prompt_text,
                    system_instruction=system_prompt,
                    model=gem_model
                ):
                    chunk = {
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": req.model,
                        "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": None}],
                    }
                    yield f"data: {json.dumps(chunk)}\n\n"
            except Exception as exc:
                err_chunk = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": req.model,
                    "choices": [{"index": 0, "delta": {"content": f"\n\n⚠️ Gemini error: {exc}"}, "finish_reason": "stop"}],
                }
                yield f"data: {json.dumps(err_chunk)}\n\n"

            final_chunk = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": req.model,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }
            yield f"data: {json.dumps(final_chunk)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(sse_gemini(), media_type="text/event-stream")

    # If we already have a connected client for "everything before this newest
    # message", reuse it and send only the newest message. Otherwise this is a
    # new (or forgotten, e.g. after a server restart) conversation — a fresh
    # client will be created, so replay the full transcript to restore context.
    pool_key = _fingerprint(history[:-1]) if len(history) >= 2 else str(uuid.uuid4())

    completion_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())

    try:
        client, is_new = await get_client(pool_key, system_prompt)
    except Exception as exc:
        err_msg = str(exc)
        if "Not logged in" in err_msg or "login" in err_msg.lower() or "auth" in err_msg.lower():
            raise HTTPException(
                status_code=401,
                detail="Chưa đăng nhập Claude CLI. Vui lòng chạy 'docker compose run --rm login' hoặc nhấp đăng nhập trên UI."
            ) from exc
        raise HTTPException(status_code=400, detail=f"Failed to start agent: {err_msg}") from exc

    # history[-1] có thể chứa ảnh (list content block) — convert riêng để KHÔNG mất ảnh, rồi
    # mới chèn transcript text của các lượt trước (nếu client vừa tạo mới, cần replay ngữ cảnh).
    last_content = _openai_content_to_user_content(history[-1].content)
    if is_new:
        history_text = _transcript(history[:-1])
        if not history_text:
            prompt: UserContent = last_content
        elif isinstance(last_content, str):
            prompt = f"{history_text}\nUser: {last_content}"
        else:
            prompt = [{"type": "text", "text": history_text}, *last_content]
    else:
        prompt = last_content

    if not req.stream:
        try:
            text_parts = [chunk async for chunk in stream_turn(client, prompt)]
        except Exception:
            await drop_client(pool_key)
            raise
        content = "".join(text_parts)
        # Re-key under "conversation + reply just given" so the next request
        # (whose history[:-1] will be exactly that) finds this same client.
        final_key = _fingerprint(history + [ChatMessage(role="assistant", content=content)])
        rekey_client(pool_key, final_key, client)
        return {
            "id": completion_id,
            "object": "chat.completion",
            "created": created,
            "model": MODEL_ID,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
        }

    async def sse():
        first_chunk = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": MODEL_ID,
            "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
        }
        yield f"data: {json.dumps(first_chunk)}\n\n"

        collected: list[str] = []
        try:
            async for text in stream_turn(client, prompt):
                collected.append(text)
                chunk = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": MODEL_ID,
                    "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": None}],
                }
                yield f"data: {json.dumps(chunk)}\n\n"
        except Exception:
            await drop_client(pool_key)
            raise

        full_content = "".join(collected)
        final_key = _fingerprint(
            history + [ChatMessage(role="assistant", content=full_content)]
        )
        rekey_client(pool_key, final_key, client)

        final_chunk = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": MODEL_ID,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }
        yield f"data: {json.dumps(final_chunk)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(sse(), media_type="text/event-stream")
