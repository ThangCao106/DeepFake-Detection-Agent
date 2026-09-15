import base64
import json
import os
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any, TypedDict

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ClaudeSDKClient,
    StreamEvent,
    TextBlock,
    ToolResultBlock,
    UserMessage,
    query,
)

from app.config import ALL_MCP_SERVERS, ALLOWED_TOOLS, DISALLOWED_TOOLS, PROJECT_ROOT


@dataclass
class _PooledClient:
    client: ClaudeSDKClient
    last_used: float = field(default_factory=time.monotonic)


# Keyed by a caller-chosen conversation key (e.g. LibreChat conversation
# fingerprint, or our own session_id). Each entry is a *connected* client whose
# underlying claude.exe + MCP subprocess are already up — reusing it instead of
# spawning fresh ones is what turns a ~15-20s per-message cost into a one-time
# cost per conversation. See README "Vì sao chậm" section.
_client_pool: dict[str, _PooledClient] = {}

# Trước đây KHÔNG có gì tự dọn: mỗi hội thoại mới mở (LibreChat/n8n) sinh ra 1 subprocess
# Claude Code + MCP riêng, và nếu user chỉ đóng tab (không có tín hiệu gì gửi về server) thì
# subprocess đó sống vô thời hạn, rò rỉ tài nguyên dần theo thời gian. Giờ tự drop sau
# 60 phút không hoạt động — an toàn vì lịch sử hội thoại (_session_history bên dưới) được
# lưu riêng, KHÔNG bị TTL này ảnh hưởng, nên khi client bị drop rồi tạo lại, /api/chat vẫn
# replay lại được lịch sử (xem prepend_history) — không mất trí nhớ, chỉ tốn thêm round-trip
# khởi động subprocess (~15-20s) cho lượt hỏi tiếp theo sau khi bị TTL.
_SESSION_TTL_SECONDS = 60 * 60


async def _evict_stale_clients() -> None:
    now = time.monotonic()
    stale_keys = [key for key, pooled in _client_pool.items() if now - pooled.last_used > _SESSION_TTL_SECONDS]
    for key in stale_keys:
        pooled = _client_pool.pop(key, None)
        if pooled is not None:
            try:
                await pooled.client.disconnect()
            except Exception:
                pass  # best-effort — subprocess có thể đã tự chết sẵn, không chặn request mới vì lỗi cleanup


class Attachment(TypedDict):
    media_type: str  # e.g. "image/png", "application/pdf", "text/plain"
    data: str  # base64-encoded, no "data:...;base64," prefix
    filename: str | None


# What we actually pass to claude_agent_sdk: either a plain string (no attachments,
# unchanged fast path) or a list of Anthropic Messages API content blocks (text +
# image/document blocks). This is the ONLY way this agent can "see" a file/image —
# it deliberately does NOT get the Read tool (see DISALLOWED_TOOLS in config.py), so
# there's no way for it to open arbitrary files on the server; content only ever
# arrives if the caller attaches it directly in the request.
UserContent = str | list[dict[str, Any]]


def build_user_content(text: str, attachments: list[Attachment] | None = None) -> UserContent:
    """Turn a text message + optional attachments into SDK-ready content.

    Images and PDFs become proper Messages API content blocks (rendered/parsed
    natively by the model). Any other file type is decoded as text and inlined
    with a header, since Claude has no generic "document" block for e.g. .txt/.csv/.log.
    """
    if not attachments:
        return text

    blocks: list[dict[str, Any]] = []
    for att in attachments:
        media_type = att["media_type"]
        label = att.get("filename") or media_type
        if media_type.startswith("image/"):
            blocks.append(
                {
                    "type": "image",
                    "source": {"type": "base64", "media_type": media_type, "data": att["data"]},
                }
            )
        elif media_type == "application/pdf":
            blocks.append(
                {
                    "type": "document",
                    "source": {"type": "base64", "media_type": media_type, "data": att["data"]},
                }
            )
        else:
            try:
                decoded = base64.b64decode(att["data"]).decode("utf-8", errors="replace")
            except Exception:
                decoded = "(không decode được nội dung file này)"
            blocks.append({"type": "text", "text": f"--- file: {label} ---\n{decoded}"})

    # Text instruction last, after any attachments — matches Anthropic's guidance to
    # place images/documents before the text that references them.
    blocks.append({"type": "text", "text": text})
    return blocks


async def _single_user_message(content: UserContent) -> AsyncIterator[dict[str, Any]]:
    yield {"type": "user", "message": {"role": "user", "content": content}}


# Lịch sử text thuần (chỉ câu trả lời cuối cùng, KHÔNG gồm phần hiển thị tool-call/thinking —
# xem stream_turn's answer_parts) cho /api/chat. Tách riêng khỏi _client_pool và KHÔNG có TTL —
# mục đích duy nhất là để replay lại khi client bị TTL/lỗi drop rồi tạo mới (xem get_client
# docstring), nên phải sống LÂU HƠN client thì mới có tác dụng.
_MAX_HISTORY_TURNS = 40  # ~20 lượt hỏi-đáp — chặn transcript phình vô hạn nếu 1 session sống rất lâu
_session_history: dict[str, list[dict[str, str]]] = {}


def record_turn(key: str, role: str, text: str) -> None:
    """Lưu 1 lượt hội thoại (role: "user"/"assistant") cho /api/chat để replay sau này."""
    if not text:
        return
    history = _session_history.setdefault(key, [])
    history.append({"role": role, "text": text})
    if len(history) > _MAX_HISTORY_TURNS:
        del history[: len(history) - _MAX_HISTORY_TURNS]


def prepend_history(content: UserContent, key: str) -> UserContent:
    """Chèn lịch sử text đã lưu (nếu có) vào TRƯỚC nội dung hiện tại — gọi khi get_client()
    trả về is_new=True (client vừa tạo mới, không có ngữ cảnh gì), để agent không "mất trí
    nhớ" hội thoại cũ. Không có gì để chèn (session mới tinh, chưa từng record_turn) thì trả
    nguyên content, không đổi gì."""
    history = _session_history.get(key)
    if not history:
        return content
    history_text = "\n".join(
        f"{'User' if h['role'] == 'user' else 'Assistant'}: {h['text']}" for h in history
    )
    if isinstance(content, str):
        return f"{history_text}\nUser: {content}"
    return [{"type": "text", "text": history_text}, *content]


def _build_options(system_prompt: str, partial: bool = False) -> ClaudeAgentOptions:
    env_dict = dict(os.environ)
    api_key = os.environ.get("CLAUDE_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")
    if api_key:
        env_dict["ANTHROPIC_API_KEY"] = api_key
        env_dict["CLAUDE_API_KEY"] = api_key

    return ClaudeAgentOptions(
        cwd=PROJECT_ROOT,
        system_prompt=system_prompt,
        mcp_servers=ALL_MCP_SERVERS,
        allowed_tools=ALLOWED_TOOLS,
        disallowed_tools=DISALLOWED_TOOLS,
        permission_mode="default",
        include_partial_messages=partial,
        thinking={"type": "adaptive", "display": "summarized"},
        env=env_dict,
    )


async def get_client(key: str, system_prompt: str) -> tuple[ClaudeSDKClient, bool]:
    """Return a connected client for this conversation key.

    Returns (client, is_new). is_new=True means this client was just created and
    connected (paid the subprocess-startup cost) — either a brand-new conversation,
    or an old one whose client got TTL/lỗi-dropped (xem _SESSION_TTL_SECONDS) — trong
    cả 2 trường hợp, client mới không có ngữ cảnh gì, caller nên replay lại lịch sử
    (openai_compat.py tự làm từ transcript LibreChat gửi; /api/chat dùng prepend_history).
    is_new=False nghĩa là client đã có sẵn ngữ cảnh — chỉ cần gửi đúng tin nhắn mới nhất.
    """
    await _evict_stale_clients()

    pooled = _client_pool.get(key)
    if pooled is not None:
        pooled.last_used = time.monotonic()
        return pooled.client, False

    options = _build_options(system_prompt, partial=True)
    client = ClaudeSDKClient(options=options)
    await client.connect()
    _client_pool[key] = _PooledClient(client)
    return client, True


async def drop_client(key: str) -> None:
    """Disconnect and forget a pooled client (e.g. on error, to force a clean retry)."""
    pooled = _client_pool.pop(key, None)
    if pooled is not None:
        await pooled.client.disconnect()


def rekey_client(old_key: str, new_key: str, client: ClaudeSDKClient) -> None:
    """Move a pooled client to a new key once we know what the next lookup will be.

    Used after a turn completes: the caller looked the client up under
    `old_key` (a temporary/uuid key for brand-new conversations), and now
    re-indexes it under `new_key` (fingerprint of the full transcript so far,
    including the reply just generated) so the next request finds it.
    """
    _client_pool.pop(old_key, None)
    _client_pool[new_key] = _PooledClient(client)


async def run_agent(prompt: UserContent, system_prompt: str) -> str:
    """One-shot helper for /api/alert — no conversation continuity needed, so a
    plain query() (spawn, run once, exit) is fine here."""
    options = _build_options(system_prompt)
    # query() only accepts str | AsyncIterable[dict] — a plain string takes the
    # fast path unchanged, a content-block list needs wrapping into a 1-item stream.
    sdk_prompt = prompt if isinstance(prompt, str) else _single_user_message(prompt)
    text_parts: list[str] = []
    async for message in query(prompt=sdk_prompt, options=options):
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if isinstance(block, TextBlock):
                    text_parts.append(block.text)
    return "\n".join(text_parts)


# Giới hạn riêng cho phần HIỂN THỊ trace (thinking + tool input/output) trong
# stream_turn — KHÔNG ảnh hưởng gì tới ngữ cảnh model thấy (model nhận tool
# result đầy đủ qua chính giao thức SDK trước khi tới bước format hiển thị này).
# Giảm mạnh (thay vì 3000) để khối trace không phình dài vô tận khi agent gọi
# nhiều tool/thinking dài — LibreChat không tự giới hạn chiều cao code block.
_TOOL_RESULT_MAX_CHARS = 400
_TOOL_INPUT_MAX_CHARS = 200
_THINKING_TRACE_MAX_CHARS = 400


def _unwrap_nested_json(value: Any, depth: int = 0) -> Any:
    """mcp-clickhouse (và nhiều MCP server khác) hay trả JSON-encode 2 lần — {"result": "{\"columns\":...}"}
    thay vì {"result": {"columns":...}}. Đệ quy parse mọi string trông giống JSON để expand ra
    thành object thật, dump lại 1 lần duy nhất sẽ hết cảnh \" chi chít khó đọc. Giới hạn depth để
    tránh loop vô hạn/tốn CPU với input độc hại."""
    if depth > 4:
        return value
    if isinstance(value, str):
        stripped = value.strip()
        if stripped[:1] in "{[":
            try:
                parsed = json.loads(stripped)
            except json.JSONDecodeError:
                return value
            return _unwrap_nested_json(parsed, depth + 1)
        return value
    if isinstance(value, dict):
        return {k: _unwrap_nested_json(v, depth + 1) for k, v in value.items()}
    if isinstance(value, list):
        return [_unwrap_nested_json(v, depth + 1) for v in value]
    return value


def _format_tool_result_content(content: str | list[dict[str, Any]] | None) -> str:
    """ToolResultBlock.content is str | list[dict] | None — flatten to displayable text,
    unwrap double-encoded JSON nếu có để dễ đọc."""
    if content is None:
        return "(không có output)"
    if isinstance(content, str):
        text = content
    else:
        text = "\n".join(
            block.get("text", "") for block in content if isinstance(block, dict) and block.get("type") == "text"
        ) or json.dumps(content, ensure_ascii=False, indent=2)

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        pass
    else:
        text = json.dumps(_unwrap_nested_json(parsed), ensure_ascii=False, indent=2)

    if len(text) > _TOOL_RESULT_MAX_CHARS:
        text = text[:_TOOL_RESULT_MAX_CHARS] + f"\n... (đã cắt bớt, còn {len(text) - _TOOL_RESULT_MAX_CHARS} ký tự)"
    return text


async def stream_turn(
    client: ClaudeSDKClient, prompt: UserContent, answer_parts: list[str] | None = None
) -> AsyncIterator[str]:
    """Send one turn to an already-connected client, yielding text as it streams.

    Surfaces thinking text and tool calls (rendered as Markdown blocks with input +
    output) in addition to the final answer text, so the caller sees continuous
    activity instead of a silent wait while e.g. a ClickHouse query runs.

    If answer_parts is given, ONLY the real answer text (text_delta / fallback
    AssistantMessage text — KHÔNG gồm thinking hay tool-call display) gets appended
    vào đó, để caller dùng cho record_turn()/prepend_history() mà không lẫn phần
    hiển thị (nếu replay nguyên cả markdown tool-call cũ vào prompt sau này sẽ vừa
    tốn token vừa rối cho model).
    """
    if isinstance(prompt, str):
        await client.query(prompt)
    else:
        await client.query(_single_user_message(prompt))
    streamed_any_delta = False

    # Trace (thinking + tool calls) được stream REALTIME như trước, nhưng gói
    # chung vào 1 fenced code block DUY NHẤT (4 dấu backtick — dài hơn các fence
    # ```json/``` lồng bên trong cho input/output, để CommonMark không hiểu nhầm
    # fence con là điểm đóng fence cha). Khung này mở ra ở event trace đầu tiên,
    # đóng lại ngay khi text_delta của câu trả lời thật bắt đầu — nên user vẫn
    # thấy input/output/thinking chảy real-time bên trong 1 khối, thay vì rải rác
    # nhiều khối Markdown riêng như bản cũ (gây khó cuộn khi có nhiều bước tool).
    # LƯU Ý: KHÔNG dùng <details>/<summary> để làm khung click-để-mở/đóng thật —
    # LibreChat sanitize HTML thô, tag đó hiện ra chữ nguyên văn thay vì render
    # (đã xác nhận trước đó), nên "phóng to/thu nhỏ" chỉ đạt được ở mức khối có
    # scroll nội bộ (nếu LibreChat áp max-height cho code block dài), không phải
    # toggle mở/đóng bằng click.
    trace_open = False

    def _open_trace() -> str:
        nonlocal trace_open
        trace_open = True
        return "\n\n🧠 **Quá trình xử lý (thinking + tool calls) — realtime:**\n\n````text\n"

    def _close_trace() -> str:
        nonlocal trace_open, thinking_shown
        trace_open = False
        thinking_shown = 0  # reset — đoạn trace kế tiếp (nếu có) được 1 lượt hiển thị mới
        return "\n````\n\n"

    # Đếm số ký tự thinking đã hiển thị trong đoạn trace ĐANG MỞ — cắt bớt khi vượt
    # _THINKING_TRACE_MAX_CHARS thay vì stream toàn bộ chuỗi suy nghĩ dài (nguyên nhân
    # chính khiến khối trace phình dài, dù đã có "display": "summarized").
    thinking_shown = 0

    # Theo dõi tool_use block đang mở theo index của content block trong stream
    # (event Anthropic API luôn kèm "index"), và map tool_use_id -> tên tool để
    # dùng lại lúc ToolResultBlock tới (ToolResultBlock chỉ có tool_use_id, không
    # có tên) — xem claude_agent_sdk types.py: ToolUseBlock/ToolResultBlock tách rời.
    open_tool_blocks: dict[int, dict[str, Any]] = {}
    tool_id_to_name: dict[str, str] = {}
    tool_inputs: dict[str, Any] = {}

    async for message in client.receive_response():
        if isinstance(message, StreamEvent):
            event = message.event
            event_type = event.get("type")
            index = event.get("index")

            if event_type == "content_block_start":
                block = event.get("content_block", {})
                if block.get("type") == "tool_use":
                    tool_id = block.get("id", "")
                    tool_name = block.get("name", "tool")
                    tool_id_to_name[tool_id] = tool_name
                    open_tool_blocks[index] = {"id": tool_id, "name": tool_name, "buffer": ""}
                    streamed_any_delta = True
                    piece = _open_trace() if not trace_open else ""
                    yield piece + f"\n🔧 Đang gọi `{tool_name}`...\n"

            elif event_type == "content_block_delta":
                delta = event.get("delta", {})
                delta_type = delta.get("type")

                if delta_type == "text_delta" and delta.get("text"):
                    streamed_any_delta = True
                    piece = _close_trace() if trace_open else ""
                    if answer_parts is not None:
                        answer_parts.append(delta["text"])
                    yield piece + delta["text"]

                elif delta_type == "thinking_delta" and delta.get("thinking"):
                    streamed_any_delta = True
                    text = delta["thinking"]
                    if thinking_shown >= _THINKING_TRACE_MAX_CHARS:
                        continue  # đã cắt bớt cho đoạn trace này, bỏ qua phần thinking dư
                    piece = _open_trace() if not trace_open else ""
                    remaining = _THINKING_TRACE_MAX_CHARS - thinking_shown
                    if len(text) > remaining:
                        text = text[:remaining] + "… (đã rút gọn)"
                    thinking_shown += len(text)
                    yield piece + text

                elif delta_type == "input_json_delta" and index in open_tool_blocks:
                    open_tool_blocks[index]["buffer"] += delta.get("partial_json", "")

            elif event_type == "content_block_stop" and index in open_tool_blocks:
                block_info = open_tool_blocks.pop(index)
                try:
                    tool_inputs[block_info["id"]] = json.loads(block_info["buffer"] or "{}")
                except json.JSONDecodeError:
                    tool_inputs[block_info["id"]] = block_info["buffer"]

        elif isinstance(message, UserMessage) and isinstance(message.content, list):
            for block in message.content:
                if isinstance(block, ToolResultBlock):
                    tool_name = tool_id_to_name.get(block.tool_use_id, "tool")
                    tool_input = tool_inputs.pop(block.tool_use_id, None)
                    # Input hiển thị dạng 1 dòng (không indent) + cắt bớt — full input model
                    # đã dùng để gọi tool xong rồi, đây chỉ là hiển thị cho user theo dõi.
                    input_str = (
                        json.dumps(tool_input, ensure_ascii=False)
                        if tool_input is not None
                        else "(không có input)"
                    )
                    if len(input_str) > _TOOL_INPUT_MAX_CHARS:
                        input_str = input_str[:_TOOL_INPUT_MAX_CHARS] + "… (đã rút gọn)"
                    output_str = _format_tool_result_content(block.content)
                    icon = "OK" if not block.is_error else "LỖI"
                    streamed_any_delta = True
                    # Đang ở TRONG 1 fenced code block (````text) nên markdown như **bold**
                    # lồng bên trong sẽ KHÔNG render (hiện chữ thô) — dùng định dạng log
                    # thuần (--- input ---/--- output ---) thay vì markdown/fence lồng ở đây,
                    # vừa tránh việc output của tool (có thể chứa ``` bên trong) phá vỡ fence cha.
                    piece = _open_trace() if not trace_open else ""
                    piece += (
                        f"\n[{icon}] tool: {tool_name}\n"
                        f"--- input ---\n{input_str}\n"
                        f"--- output ---\n{output_str}\n"
                    )
                    yield piece

        elif isinstance(message, AssistantMessage) and not streamed_any_delta:
            # Fallback: no partial deltas arrived for this turn (shouldn't normally
            # happen with include_partial_messages=True), emit the full text at once.
            for block in message.content:
                if isinstance(block, TextBlock) and block.text:
                    piece = _close_trace() if trace_open else ""
                    if answer_parts is not None:
                        answer_parts.append(block.text)
                    yield piece + block.text

    # Khung trace còn mở dở cuối cùng (VD: turn kết thúc chỉ với tool calls, không
    # có text_delta/AssistantMessage nào theo sau) — đóng nốt để không lộ fence hở.
    if trace_open:
        yield _close_trace()
