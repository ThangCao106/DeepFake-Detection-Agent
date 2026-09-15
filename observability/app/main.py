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


def _has_valid_credentials() -> bool:
    """Check if actual API keys or OAuth authentication credentials exist."""
    has_api_key = bool(os.environ.get("CLAUDE_API_KEY") or os.environ.get("ANTHROPIC_API_KEY"))
    if has_api_key:
        return True

    # Check if login logs confirm successful login
    global _LOGIN_LOGS
    log_text = "\n".join(_LOGIN_LOGS).lower()
    if "not logged in" not in log_text and "please run /login" not in log_text:
        if any(k in log_text for k in [
            "logged in as ", "successfully logged in", "authentication successful",
            "login successful", "welcome to claude", "authenticated as ", "successfully authenticated"
        ]):
            return True

    home_candidates = [
        Path("/home/agent"),
        Path.home(),
        Path(os.environ.get("HOME", "/home/agent"))
    ]

    for h in home_candidates:
        # Check credentials files for non-empty auth fields
        for cred_name in ["credentials.json", ".credentials.json", "session.json", "auth.json", "keys.json"]:
            for cred_path in [h / ".claude" / cred_name, h / cred_name]:
                try:
                    if cred_path.exists():
                        content = cred_path.read_text(encoding="utf-8").strip()
                        if content:
                            data = json.loads(content)
                            if isinstance(data, dict):
                                for key in ["oauthAccount", "sessionKey", "oauthToken", "primaryApiKey", "authToken", "sessionToken", "accessToken", "access_token", "accountUuid"]:
                                    if key in data and data[key]:
                                        return True
                except Exception:
                    pass

        # Check config json files for actual OAuth auth fields
        for json_path in [
            h / ".claude.json",
            h / ".claude" / "config.json",
            h / ".config" / "claude-code" / "config.json",
            h / ".config" / "claude" / "config.json"
        ]:
            try:
                if json_path.exists():
                    content = json_path.read_text(encoding="utf-8").strip()
                    if content:
                        data = json.loads(content)
                        if isinstance(data, dict):
                            for key in ["oauthAccount", "sessionKey", "oauthToken", "primaryApiKey", "authToken", "sessionToken", "accessToken", "session", "account", "userID", "email", "access_token"]:
                                if key in data and data[key]:
                                    return True
            except Exception:
                pass

    return False


@app.get("/api/auth/status")
async def auth_status() -> dict:
    """Return Claude CLI and Gemini API authentication status."""
    is_authenticated = _has_valid_credentials()
    gemini_key = os.environ.get("GEMINI_API_KEY", "").strip()
    has_gemini = bool(gemini_key)
    gemini_masked = (gemini_key[:8] + "...") if len(gemini_key) > 8 else ("***" if has_gemini else "")

    claude_key = os.environ.get("CLAUDE_API_KEY", "").strip() or os.environ.get("ANTHROPIC_API_KEY", "").strip()
    claude_masked = (claude_key[:8] + "...") if len(claude_key) > 8 else ("***" if claude_key else "")

    return {
        "authenticated": is_authenticated,
        "status": "AUTHENTICATED" if is_authenticated else "NOT_LOGGED_IN",
        "gemini_authenticated": has_gemini,
        "gemini_status": "AUTHENTICATED" if has_gemini else "NOT_CONFIGURED",
        "gemini_key_masked": gemini_masked,
        "claude_key_masked": claude_masked,
        "default_key": API_KEY,
        "login_command": "docker compose run --rm login",
        "cli_command": "npx @anthropic-ai/claude-code login",
    }


_ACTIVE_LOGIN_PROC = None
_LAST_OAUTH_URL = None
_LOGIN_LOGS = []


def _strip_ansi(text: str) -> str:
    # Strip OSC escape sequences like \x1b]8;;url\x1b\ or \x1b]...\x07
    text = re.sub(r'\x1B\][^\x07\x1B]*[\x07|\x1B\\]', '', text)
    # Strip CSI escape sequences like \x1b[31m
    text = re.sub(r'\x1B\[[0-?]*[ -/]*[@-~]', '', text)
    # Strip general escape sequences
    text = re.sub(r'\x1B[@-Z\\-_]', '', text)
    return text


def _extract_oauth_url(text: str) -> str | None:
    if not text:
        return None
    clean = _strip_ansi(text)
    
    # Priority 1: Match exact Anthropic/Claude OAuth authorization URL
    match = re.search(r'(https://(?:claude\.ai|console\.anthropic\.com)/[^\s"\'\`<>\)\(\]\[]+)', clean, re.IGNORECASE)
    if match:
        url = match.group(1).rstrip('.,;:;\\')
        if "claude.com/cai" in url:
            url = url.replace("claude.com/cai", "claude.ai")
        return url

    # Priority 2: Match any http/https URL containing anthropic/claude/oauth/authorize
    matches = re.findall(r'(https?://[^\s"\'\`<>\)\(\]\[]+)', clean, re.IGNORECASE)
    for url in matches:
        url_clean = url.rstrip('.,;:;\\')
        url_lower = url_clean.lower()
        if any(kw in url_lower for kw in ["claude", "anthropic", "oauth", "authorize"]):
            if "claude.com/cai" in url_lower:
                url_clean = url_clean.replace("claude.com/cai", "claude.ai")
            elif "claude.com" in url_lower and "oauth" in url_lower:
                url_clean = url_clean.replace("claude.com", "claude.ai")
            return url_clean
            
    return None


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
            raw_text = chunk.decode("utf-8", errors="ignore")
            clean_text = _strip_ansi(raw_text)
            buffer += clean_text

            # Extract lines from buffer
            lines = buffer.splitlines()
            for line in lines:
                clean_line = line.strip()
                if clean_line:
                    if not _LOGIN_LOGS or _LOGIN_LOGS[-1] != clean_line:
                        _LOGIN_LOGS.append(clean_line)
                    
                    found_url = _extract_oauth_url(clean_line)
                    if found_url and not _LAST_OAUTH_URL:
                        _LAST_OAUTH_URL = found_url

            # Space-collapsed matching to handle ANSI stripped terminal text
            buffer_nospace = re.sub(r'\s+', '', buffer.lower())

            # Auto-respond with Enter (\r\n) if Claude CLI pauses at initial Theme selection screen
            if any(k in buffer_nospace for k in ["choosethetextstyle", "letsgetstarted", "monokaiextended", "textstylethatlooksbest"]):
                if proc.stdin:
                    try:
                        proc.stdin.write(b"\r\n")
                        proc.stdin.write(b"\r")
                        await proc.stdin.drain()
                        if not getattr(proc, "_auto_entered_theme", False):
                            proc._auto_entered_theme = True
                            _LOGIN_LOGS.append("➡️ [AUTO-ENTER] Sent carriage return (\\r\\n) to select default Dark mode theme.")
                    except Exception as e_stdin:
                        _LOGIN_LOGS.append(f"⚠️ Warning sending auto-enter: {e_stdin}")

            # Auto-respond with /login if Claude CLI is waiting at 'Not logged in · Run /login' prompt
            if any(k in buffer_nospace for k in ["notloggedin", "run/login", "pleaserun/login"]):
                if proc.stdin and not getattr(proc, "_auto_sent_slash_login", False):
                    try:
                        proc.stdin.write(b"/login\r\n")
                        await proc.stdin.drain()
                        proc._auto_sent_slash_login = True
                        _LOGIN_LOGS.append("➡️ [AUTO-ENTER] Automatically sent '/login\\r\\n' command to Claude CLI REPL.")
                    except Exception as e_stdin:
                        _LOGIN_LOGS.append(f"⚠️ Warning sending /login command: {e_stdin}")

            # Auto-respond to Workspace Safety Check (select '2. Yes, I trust this folder')
            if any(k in buffer_nospace for k in ["quicksafetycheck", "isthisaproject", "trustthisfolder", "securityguide", "entertoconfirm"]):
                if proc.stdin:
                    try:
                        if not getattr(proc, "_auto_approved_safety", False):
                            proc.stdin.write(b"\x1b[B\r\n")
                            proc.stdin.write(b"2\r\n")
                            await proc.stdin.drain()
                            proc._auto_approved_safety = True
                            _LOGIN_LOGS.append("➡️ [AUTO-ENTER] Automatically selected '2. Yes, I trust this folder' for workspace safety check.")
                    except Exception as e_stdin:
                        _LOGIN_LOGS.append(f"⚠️ Warning sending workspace trust selection: {e_stdin}")

            # Also check raw buffer directly for OAuth URLs
            raw_matches = re.findall(r'(https?://[^\s"\'\`<>\)\(\]\[]+)', buffer)
            for url in raw_matches:
                url_clean = url.rstrip('.,;:')
                url_lower = url_clean.lower()
                if any(kw in url_lower for kw in ["claude", "anthropic", "oauth", "authorize", "login", "account"]):
                    if not _LAST_OAUTH_URL:
                        _LAST_OAUTH_URL = url_clean

    except Exception as e:
        _LOGIN_LOGS.append(f"Output reader error: {e}")


class SetApiKeyRequest(BaseModel):
    claude_api_key: str


@app.post("/api/auth/set-key")
async def set_claude_api_key(req: SetApiKeyRequest) -> dict:
    """Dynamically set CLAUDE_API_KEY environment variable and persist to .env."""
    key = req.claude_api_key.strip()
    if not key:
        raise HTTPException(status_code=400, detail="Vui lòng nhập API Key hợp lệ.")

    os.environ["CLAUDE_API_KEY"] = key
    os.environ["ANTHROPIC_API_KEY"] = key

    # Persist to .env file in project root
    env_file = Path(PROJECT_ROOT) / ".env"
    try:
        env_lines = []
        if env_file.exists():
            env_lines = env_file.read_text(encoding="utf-8").splitlines()

        new_lines = [
            line for line in env_lines
            if not line.startswith("CLAUDE_API_KEY=") and not line.startswith("ANTHROPIC_API_KEY=")
        ]
        new_lines.append(f"CLAUDE_API_KEY={key}")
        new_lines.append(f"ANTHROPIC_API_KEY={key}")
        env_file.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
    except Exception as e:
        _LOGIN_LOGS.append(f"⚠️ Warning saving .env: {e}")

    _LOGIN_LOGS.append("✅ [SUCCESS] CLAUDE_API_KEY updated dynamically in environment!")
    return {
        "status": "SUCCESS",
        "message": "Đã lưu CLAUDE_API_KEY thành công vào hệ thống!",
        "authenticated": True,
    }


class SetGeminiKeyRequest(BaseModel):
    gemini_api_key: str


@app.post("/api/auth/set-gemini-key")
async def set_gemini_api_key(req: SetGeminiKeyRequest) -> dict:
    """Dynamically set GEMINI_API_KEY environment variable and persist to .env."""
    key = req.gemini_api_key.strip()
    if not key:
        raise HTTPException(status_code=400, detail="Vui lòng nhập Gemini API Key hợp lệ.")

    os.environ["GEMINI_API_KEY"] = key

    # Persist to .env file in project root
    env_file = Path(PROJECT_ROOT) / ".env"
class SendInputRequest(BaseModel):
    input_text: str


@app.post("/api/auth/send-input")
async def send_input_to_proc(req: SendInputRequest) -> dict:
    """Send interactive keyboard inputs (e.g. down_enter, enter, '2\n') to running login process stdin."""
    global _ACTIVE_LOGIN_PROC, _LOGIN_LOGS
    if _ACTIVE_LOGIN_PROC is None or _ACTIVE_LOGIN_PROC.returncode is not None:
        raise HTTPException(status_code=400, detail="Không có tiến trình login nào đang chạy.")

    if _ACTIVE_LOGIN_PROC.stdin:
        try:
            inp = req.input_text.strip()
            if inp == "down_enter":
                _ACTIVE_LOGIN_PROC.stdin.write(b"\x1b[B\r\n2\r\n")
            elif inp == "enter":
                _ACTIVE_LOGIN_PROC.stdin.write(b"\r\n")
            else:
                _ACTIVE_LOGIN_PROC.stdin.write(f"{inp}\n".encode("utf-8"))
            await _ACTIVE_LOGIN_PROC.stdin.drain()
            _LOGIN_LOGS.append(f"➡️ [MANUAL-INPUT] Sent input '{inp}' to process stdin")
            return {
                "status": "SUCCESS",
                "message": f"Đã gửi lệnh '{inp}' tới tiến trình!",
                "logs": _LOGIN_LOGS[-25:]
            }
        except Exception as e_inp:
            raise HTTPException(status_code=500, detail=f"Lỗi gửi input: {e_inp}")
    raise HTTPException(status_code=500, detail="Process stdin is not writable.")


class SubmitAuthCodeRequest(BaseModel):
    code: str


@app.post("/api/auth/submit-code")
async def submit_auth_code(req: SubmitAuthCodeRequest) -> dict:
    """Submit OAuth authorization code or API key to running claude login subprocess stdin or environment."""
    global _ACTIVE_LOGIN_PROC, _LOGIN_LOGS
    code_input = req.code.strip().strip("'\"")
    if not code_input:
        raise HTTPException(status_code=400, detail="Vui lòng nhập mã code hoặc API Key hợp lệ.")

    # 1. If user pasted an Anthropic API Key (sk-ant-...)
    if code_input.startswith("sk-ant-"):
        os.environ["CLAUDE_API_KEY"] = code_input
        os.environ["ANTHROPIC_API_KEY"] = code_input
        env_file = Path(PROJECT_ROOT) / ".env"
        try:
            env_lines = env_file.read_text(encoding="utf-8").splitlines() if env_file.exists() else []
            new_lines = [l for l in env_lines if not l.startswith("CLAUDE_API_KEY=") and not l.startswith("ANTHROPIC_API_KEY=")]
            new_lines.append(f"CLAUDE_API_KEY={code_input}")
            new_lines.append(f"ANTHROPIC_API_KEY={code_input}")
            env_file.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
        except Exception as e:
            _LOGIN_LOGS.append(f"⚠️ Warning saving .env: {e}")

        _LOGIN_LOGS.append("✅ [SUCCESS] Direct Anthropic API Key saved from Submit Code input.")
        return {
            "status": "SUCCESS",
            "message": "Đã tự động lưu Claude API Key vào hệ thống!",
            "authenticated": True,
            "logs": _LOGIN_LOGS[-25:],
        }

    # 2. OAuth Code: If login process isn't active, launch fresh login process
    if _ACTIVE_LOGIN_PROC is None or _ACTIVE_LOGIN_PROC.returncode is not None:
        _LOGIN_LOGS.append("🔄 [SUBMIT-CODE] Launching Claude CLI login process for OAuth code exchange...")
        cmd = "claude login" if shutil.which("claude") else "npx -y @anthropic-ai/claude-code login"
        if shutil.which("script"):
            cmd = f'script -q -c "{cmd}" /dev/null'
        
        env = os.environ.copy()
        env.pop("CI", None)
        env["FORCE_COLOR"] = "0"
        env["HOME"] = "/home/agent"
        env["DISABLE_CLAUDE_AUTO_UPDATE"] = "1"
        env["DISABLE_AUTO_UPDATE"] = "1"
        env["CLAUDE_AUTO_UPDATE_DISABLE"] = "1"
        env["NO_UPDATE_NOTIFIER"] = "1"

        try:
            _ACTIVE_LOGIN_PROC = await asyncio.create_subprocess_shell(
                cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                cwd="/tmp",
                env=env
            )
            asyncio.create_task(_read_subprocess_output(_ACTIVE_LOGIN_PROC))
            await asyncio.sleep(1.5)
        except Exception as proc_exc:
            _LOGIN_LOGS.append(f"❌ [SUBMIT-CODE ERROR] Failed to start login process: {proc_exc}")
            raise HTTPException(status_code=500, detail=f"Không thể khởi chạy tiến trình login: {proc_exc}")

    # 3. Write code to process stdin ONCE
    if _ACTIVE_LOGIN_PROC and _ACTIVE_LOGIN_PROC.stdin:
        try:
            _ACTIVE_LOGIN_PROC.stdin.write(f"{code_input}\n".encode("utf-8"))
            await _ACTIVE_LOGIN_PROC.stdin.drain()
            _LOGIN_LOGS.append(f"➡️ [SUBMIT-CODE] Sent OAuth Code ({code_input[:8]}...{len(code_input)} chars) to Claude CLI stdin.")
            
            await asyncio.sleep(1.2)
            return {
                "status": "SUCCESS",
                "message": "Đã dán mã xác thực OAuth tới tiến trình CLI! Đang thực hiện đăng nhập...",
                "authenticated": False,
                "logs": _LOGIN_LOGS[-25:],
            }
        except Exception as exc:
            _LOGIN_LOGS.append(f"⚠️ Warning writing stdin: {exc}")
            raise HTTPException(status_code=500, detail=f"Lỗi ghi mã vào stdin: {exc}")

    raise HTTPException(status_code=500, detail="Tiến trình login CLI không sẵn sàng ghi stdin.")


@app.get("/api/auth/debug-logs")
async def get_debug_logs() -> dict:
    """Return full debug trace and authentication state diagnostics."""
    global _ACTIVE_LOGIN_PROC, _LAST_OAUTH_URL, _LOGIN_LOGS
    proc_status = "NOT_STARTED"
    if _ACTIVE_LOGIN_PROC is not None:
        if _ACTIVE_LOGIN_PROC.returncode is None:
            proc_status = f"RUNNING (PID: {_ACTIVE_LOGIN_PROC.pid})"
        else:
            proc_status = f"EXITED (ReturnCode: {_ACTIVE_LOGIN_PROC.returncode})"

    home_dir = os.environ.get("HOME", "/home/agent")
    claude_dir = Path(home_dir) / ".claude"
    user_claude_dir = Path.home() / ".claude"

    return {
        "process_status": proc_status,
        "oauth_url": _LAST_OAUTH_URL,
        "claude_dir_exists": claude_dir.exists(),
        "user_claude_dir_exists": user_claude_dir.exists(),
        "has_api_key": bool(os.environ.get("CLAUDE_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")),
        "total_log_lines": len(_LOGIN_LOGS),
        "logs": _LOGIN_LOGS,
    }


@app.post("/api/auth/trigger-login")
async def trigger_login() -> dict:
    """Auto-launch Claude CLI login subprocess non-blockingly and capture OAuth URL."""
    global _ACTIVE_LOGIN_PROC, _LAST_OAUTH_URL, _LOGIN_LOGS
    import asyncio
    import os
    import re
    import shutil

    # Smart command selection based on available executables in environment.
    # Wraps in 'script -q -c ... /dev/null' if available to allocate a pseudo-TTY for non-interactive containers,
    # preventing Claude CLI from failing with 'Warning: no stdin data received in 3s'.
    if shutil.which("claude"):
        base_cmd = "claude login"
    elif shutil.which("npx"):
        base_cmd = "npx -y @anthropic-ai/claude-code login"
    elif shutil.which("docker"):
        base_cmd = "docker compose run --rm login"
    else:
        base_cmd = "npx -y @anthropic-ai/claude-code login"

    if shutil.which("script") and not base_cmd.startswith("docker"):
        cmd = f'script -q -c "{base_cmd}" /dev/null'
    else:
        cmd = base_cmd

    # Return cached URL if available
    if _LAST_OAUTH_URL:
        return {
            "status": "SUCCESS",
            "message": "OAuth URL retrieved from cache.",
            "command": cmd,
            "oauth_url": _LAST_OAUTH_URL,
            "logs": _LOGIN_LOGS[-25:],
        }

    try:
        if _ACTIVE_LOGIN_PROC is None or _ACTIVE_LOGIN_PROC.returncode is not None:
            _LOGIN_LOGS = []
            env = os.environ.copy()
            env.pop("CI", None)
            env["FORCE_COLOR"] = "0"
            env["HOME"] = "/home/agent"
            env["DISABLE_CLAUDE_AUTO_UPDATE"] = "1"
            env["DISABLE_AUTO_UPDATE"] = "1"
            env["CLAUDE_AUTO_UPDATE_DISABLE"] = "1"
            env["NO_UPDATE_NOTIFIER"] = "1"

            # Pre-create default theme config in all Claude CLI config locations across all candidate home directories
            try:
                cfg_data = {
                    "theme": "dark",
                    "hasCompletedOnboarding": True,
                    "hasSeenWelcome": True,
                    "hasCompletedTour": True,
                    "trustedDirectories": ["/app", "/home/agent", "/root", "."],
                    "projects": {
                        "/app": {"hasCompletedOnboarding": True, "trustState": "TRUSTED"},
                        "/home/agent": {"hasCompletedOnboarding": True, "trustState": "TRUSTED"},
                        "/root": {"hasCompletedOnboarding": True, "trustState": "TRUSTED"}
                    }
                }
                cfg_payload = json.dumps(cfg_data, indent=2)
                
                home_candidates = [
                    Path("/home/agent"),
                    Path.home(),
                    Path("/root"),
                    Path(env.get("HOME", "/home/agent"))
                ]
                
                for h_dir in home_candidates:
                    try:
                        h_dir.mkdir(parents=True, exist_ok=True)
                        (h_dir / ".claude.json").write_text(cfg_payload, encoding="utf-8")
                        
                        c_sub = h_dir / ".claude"
                        c_sub.mkdir(parents=True, exist_ok=True)
                        (c_sub / "config.json").write_text(cfg_payload, encoding="utf-8")
                        
                        cfg_sub = h_dir / ".config" / "claude-code"
                        cfg_sub.mkdir(parents=True, exist_ok=True)
                        (cfg_sub / "config.json").write_text(cfg_payload, encoding="utf-8")
                    except Exception as e_h:
                        pass
            except Exception as e_cfg:
                _LOGIN_LOGS.append(f"⚠️ Warning pre-creating .claude.json: {e_cfg}")

            _LOGIN_LOGS.append(f"🚀 [DEBUG] Launching login command: '{cmd}'")
            _LOGIN_LOGS.append(f"📂 [DEBUG] Login Working Dir: '/tmp', HOME: '{env.get('HOME')}'")
            _LOGIN_LOGS.append(f"🔑 [DEBUG] API Key present in env: {bool(env.get('CLAUDE_API_KEY') or env.get('ANTHROPIC_API_KEY'))}")

            _ACTIVE_LOGIN_PROC = await asyncio.create_subprocess_shell(
                cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                cwd="/tmp",
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
                found_url = _extract_oauth_url(log)
                if found_url:
                    _LAST_OAUTH_URL = found_url
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
            "logs": _LOGIN_LOGS[-25:],
        }
    except Exception as exc:
        _LOGIN_LOGS.append(f"❌ [DEBUG ERROR] Execution error: {exc}")
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
            content = prepend_history(content, req.session_id)
        answer_parts: list[str] = []
        text_parts = [chunk async for chunk in stream_turn(client, content, answer_parts=answer_parts)]
        full_reply = "".join(text_parts)
        if any(k in full_reply for k in ["Invalid API key", "Fix external API key", "invalid api key"]):
            await drop_client(req.session_id)
            os.environ.pop("CLAUDE_API_KEY", None)
            os.environ.pop("ANTHROPIC_API_KEY", None)
            raise HTTPException(
                status_code=401,
                detail="Lỗi API Key không hợp lệ. Hệ thống đã tự động dọn dẹp Key hỏng. Vui lòng thử lại hoặc bấm '🔑 Claude Login' để dùng phiên OAuth."
            )
        if any(k in full_reply for k in ["Not logged in", "Please run /login", "run /login"]):
            await drop_client(req.session_id)
            raise HTTPException(
                status_code=401,
                detail="Chưa hoàn tất đăng nhập Claude CLI. Vui lòng nhấn nút '🔑 Claude Login' trên giao diện để mở trang xác thực Anthropic OAuth hoặc nhập Claude API Key."
            )
    except Exception as exc:
        await drop_client(req.session_id)
        if isinstance(exc, HTTPException):
            raise exc
        err_msg = str(exc)
        if "Not logged in" in err_msg or "login" in err_msg.lower() or "auth" in err_msg.lower():
            raise HTTPException(
                status_code=401,
                detail="Chưa hoàn tất đăng nhập Claude CLI. Vui lòng nhấn nút '🔑 Claude Login' trên giao diện để mở liên kết đăng nhập OAuth."
            ) from exc
        raise HTTPException(status_code=400, detail=f"Agent error: {err_msg}") from exc
    record_turn(req.session_id, "user", req.message)
    record_turn(req.session_id, "assistant", full_reply)
    return {"reply": full_reply}


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
