import json
import os
import re
import subprocess
from pathlib import Path

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.agent import (
    build_user_content,
    drop_client,
    get_client,
    prepend_history,
    record_turn,
    run_agent,
    stream_turn,
)
from app.config import ALERT_SYSTEM_PROMPT, API_KEY, CHAT_SYSTEM_PROMPT, PROJECT_ROOT
from app.openai_compat import router as openai_router

app = FastAPI(title="Observability Agent (local test)")
app.include_router(openai_router)


class GateLoginRequest(BaseModel):
    username: str
    password: str


@app.post("/api/auth/gate-login")
async def gate_login(req: GateLoginRequest) -> dict:
    """Verify user login credentials for Obsagent Subsystem (thang.cd / Thang@123)."""
    if req.username == "thang.cd" and req.password == "Thang@123":
        return {
            "status": "SUCCESS",
            "authenticated": True,
            "user": req.username,
            "message": "Subsystem access granted.",
        }
    raise HTTPException(status_code=401, detail="Tên đăng nhập hoặc mật khẩu không chính xác")


@app.get("/api/auth/status")
async def auth_status() -> dict:
    """Return Claude CLI authentication status and login instructions."""
    home_dir = os.environ.get("HOME", "/home/agent")
    claude_dir = Path(home_dir) / ".claude"
    user_claude_dir = Path.home() / ".claude"

    is_authenticated = claude_dir.exists() or user_claude_dir.exists()
    return {
        "authenticated": is_authenticated,
        "status": "AUTHENTICATED" if is_authenticated else "NOT_LOGGED_IN",
        "default_key": API_KEY,
        "login_command": "docker compose run --rm login",
        "cli_command": "npx @anthropic-ai/claude-code login",
    }


_ACTIVE_LOGIN_PROC = None
_LAST_OAUTH_URL = None
_LOGIN_LOGS = []


async def _read_subprocess_output(proc):
    """Background task to continuously read subprocess stdout asynchronously without blocking event loop."""
    global _LAST_OAUTH_URL, _LOGIN_LOGS
    import re
    buffer = ""
    try:
        while True:
            chunk = await proc.stdout.read(256)
            if not chunk:
                break
            text = chunk.decode("utf-8", errors="ignore")
            buffer += text

            # Extract lines from buffer
            lines = buffer.splitlines()
            for line in lines:
                clean_line = line.strip()
                if clean_line:
                    if not _LOGIN_LOGS or _LOGIN_LOGS[-1] != clean_line:
                        _LOGIN_LOGS.append(clean_line)
                    match = re.search(
                        r'(https?://[^\s"\'\`]+(?:claude|anthropic|oauth|authorize)[^\s"\'\`]*)',
                        clean_line,
                        re.IGNORECASE,
                    )
                    if match and not _LAST_OAUTH_URL:
                        _LAST_OAUTH_URL = match.group(1)

            # Also check raw text chunk directly for OAuth URLs
            match_raw = re.search(
                r'(https?://[^\s"\'\`]+(?:claude|anthropic|oauth|authorize)[^\s"\'\`]*)',
                buffer,
                re.IGNORECASE,
            )
            if match_raw and not _LAST_OAUTH_URL:
                _LAST_OAUTH_URL = match_raw.group(1)

    except Exception as e:
        _LOGIN_LOGS.append(f"Output reader error: {e}")


@app.post("/api/auth/trigger-login")
async def trigger_login() -> dict:
    """Auto-launch Claude CLI login subprocess non-blockingly and capture OAuth URL."""
    global _ACTIVE_LOGIN_PROC, _LAST_OAUTH_URL, _LOGIN_LOGS
    import asyncio
    import os
    import re
    import shutil

    # Smart command selection based on available executables in environment
    if shutil.which("claude"):
        cmd = "claude login"
    elif shutil.which("npx"):
        cmd = "npx -y @anthropic-ai/claude-code login"
    elif shutil.which("docker"):
        cmd = "docker compose run --rm login"
    else:
        cmd = "npx -y @anthropic-ai/claude-code login"

    # Return cached URL if available
    if _LAST_OAUTH_URL:
        return {
            "status": "SUCCESS",
            "message": "OAuth URL retrieved from cache.",
            "command": cmd,
            "oauth_url": _LAST_OAUTH_URL,
            "logs": _LOGIN_LOGS[-20:],
        }

    try:
        if _ACTIVE_LOGIN_PROC is None or _ACTIVE_LOGIN_PROC.returncode is not None:
            _LOGIN_LOGS = []
            env = os.environ.copy()
            env["CI"] = "1"
            env["FORCE_COLOR"] = "0"
            env["HOME"] = "/home/agent"

            _ACTIVE_LOGIN_PROC = await asyncio.create_subprocess_shell(
                cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                cwd=PROJECT_ROOT,
                env=env
            )
            asyncio.create_task(_read_subprocess_output(_ACTIVE_LOGIN_PROC))

        # Non-blocking poll for up to 3 seconds for initial fast detection
        for _ in range(6):
            if _LAST_OAUTH_URL:
                break
            await asyncio.sleep(0.5)

        # Fallback regex check over captured logs
        if not _LAST_OAUTH_URL:
            for log in _LOGIN_LOGS:
                match = re.search(
                    r'(https?://[^\s"\'\`]+(?:claude|anthropic|oauth|authorize)[^\s"\'\`]*)',
                    log,
                    re.IGNORECASE,
                )
                if match:
                    _LAST_OAUTH_URL = match.group(1)
                    break

        return {
            "status": "SUCCESS" if _LAST_OAUTH_URL else "PENDING",
            "message": (
                "OAuth URL captured successfully!"
                if _LAST_OAUTH_URL
                else f"Login command '{cmd}' running..."
            ),
            "command": cmd,
            "oauth_url": _LAST_OAUTH_URL,
            "logs": _LOGIN_LOGS[-20:],
        }
    except Exception as exc:
        return {
            "status": "ERROR",
            "message": f"Execution error: {exc}",
            "command": cmd,
            "oauth_url": None,
            "logs": _LOGIN_LOGS,
        }


def check_api_key(x_api_key: str = Header(...)) -> None:
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="invalid api key")


class ChatAttachment(BaseModel):
    media_type: str  # "image/png", "image/jpeg", "application/pdf", "text/plain", ...
    data: str  # base64, KHÔNG kèm prefix "data:...;base64,"
    filename: str | None = None


class ChatRequest(BaseModel):
    session_id: str
    message: str
    attachments: list[ChatAttachment] = []


@app.post("/api/chat", dependencies=[Depends(check_api_key)])
async def chat(req: ChatRequest) -> dict:
    # session_id here is a stable key from the caller (unlike LibreChat's
    # OpenAI-compat path, which has to fingerprint the transcript instead —
    # see app/openai_compat.py). Reusing the connected client per session_id
    # is what avoids paying the ~15-20s subprocess-startup cost every message.
    content = build_user_content(
        req.message, [att.model_dump() for att in req.attachments]
    )
    try:
        client, is_new = await get_client(req.session_id, CHAT_SYSTEM_PROMPT)
        if is_new:
            # Client vừa được tạo mới (session đầu tiên, hoặc client cũ đã bị TTL/lỗi
            # drop — xem agent.py _SESSION_TTL_SECONDS) — không có ngữ cảnh gì, phải
            # chèn lại lịch sử đã lưu trước đó để agent không "mất trí nhớ" hội thoại cũ.
            content = prepend_history(content, req.session_id)
        answer_parts: list[str] = []
        text_parts = [chunk async for chunk in stream_turn(client, content, answer_parts=answer_parts)]
    except Exception as exc:
        await drop_client(req.session_id)
        err_msg = str(exc)
        if "Not logged in" in err_msg or "login" in err_msg.lower() or "auth" in err_msg.lower():
            raise HTTPException(
                status_code=401,
                detail="Chưa đăng nhập Claude CLI. Vui lòng nhấn nút '🔑 Claude Login' trên giao diện để mở liên kết đăng nhập OAuth."
            ) from exc
        raise HTTPException(status_code=400, detail=f"Agent error: {err_msg}") from exc
    record_turn(req.session_id, "user", req.message)
    record_turn(req.session_id, "assistant", "".join(answer_parts))
    return {"reply": "".join(text_parts)}


@app.post("/api/alert", dependencies=[Depends(check_api_key)])
async def alert(payload: dict) -> dict:
    prompt = (
        "Investigate this alert and respond with the required JSON only:\n"
        f"{json.dumps(payload, ensure_ascii=False)}"
    )
    reply = await run_agent(prompt=prompt, system_prompt=ALERT_SYSTEM_PROMPT)
    try:
        triage = json.loads(reply)
    except json.JSONDecodeError:
        return {"error": "model did not return valid JSON", "raw": reply}
    return {"triage": triage}


app.mount("/", StaticFiles(directory="static", html=True), name="static")
