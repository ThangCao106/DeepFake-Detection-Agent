# Observability Agent (bản Ubuntu)

Agent quan sát hệ thống (SRE/observability) kết nối trực tiếp vào ClickHouse, chạy trên
**Claude Agent SDK** (dùng quota Claude Pro/Max qua Claude Code login, không cần API key
trả phí riêng). Có thể chat trực tiếp qua trang web đơn giản, hoặc qua LibreChat như một
"model" bình thường.

> Đây là bản clone từ phiên bản gốc chạy trên Windows (`D:\hsc\agent\observability-agent`),
> đã chuyển các script/đường dẫn sang tương thích Linux để triển khai trên server Ubuntu.
> Code Python (`app/`) không đổi — chỉ khác script khởi động và vài đường dẫn.

> ⚠️ **Chỉ chạy local (127.0.0.1), chỉ tự dùng.** Xem mục [Rủi ro điều khoản sử dụng](#rủi-ro-điều-khoản-sử-dụng)
> trước khi mở rộng cho người khác dùng.

---

## Cài đặt trên server Ubuntu — làm theo đúng thứ tự

### 1. Chuyển code sang server

```bash
# Từ máy Windows: scp cả thư mục này sang server, hoặc git clone nếu đã push lên remote
scp -r observability-agent-ubuntu user@<server-ip>:~/observability-agent
# hoặc: git clone <remote-url> ~/observability-agent
```

### 2. Cài `uv` (chưa cần Python riêng, `uv` tự tải Python)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env   # hoặc mở lại terminal
```

### 3. Cài dependencies

```bash
cd ~/observability-agent
uv sync
```
Bước này tự tải bản `claude.exe` tương ứng cho Linux (binary khác hoàn toàn bản Windows, nhưng
package `claude-agent-sdk` tự chọn đúng theo hệ điều hành — không cần làm gì thêm).

### 4. ⚠️ QUAN TRỌNG NHẤT — auth Claude Code riêng cho server này

**Credential đã login trên máy Windows KHÔNG tự động có trên server Ubuntu.** Đây là máy khác,
phải đăng nhập lại từ đầu. Cài Claude Code CLI rồi login:

```bash
# Cài Claude Code CLI (npm) — hoặc dùng ant CLI nếu đã quen (xem docs Anthropic)
npm install -g @anthropic-ai/claude-code
claude login
```

Nếu server **không có trình duyệt/GUI** (đa số server Ubuntu là headless), lệnh `claude login`
sẽ in ra 1 URL — mở URL đó trên máy có trình duyệt (VD máy Windows của anh), đăng nhập, quay lại
paste code xác nhận vào terminal server. Xem chi tiết flow trong tài liệu Anthropic nếu lệnh trên
không hỗ trợ `--no-browser` sẵn.

Kiểm tra login thành công:
```bash
uv run smoke_test.py
```
Nếu ra danh sách database ClickHouse thật (không lỗi `CLINotFoundError`/auth) → auth OK, sẵn sàng chạy.

### 5. Kiểm tra mạng tới ClickHouse

Server Ubuntu này **phải tới được** `CLICKHOUSE_HOST` khai trong `app/config.py`
(mặc định `192.168.53.251:8123`, mạng nội bộ HSC). Nếu server nằm ở mạng khác (VD EC2/cloud) —
**gần như chắc chắn không tới được** trừ khi có VPN/peering. Test trước bằng:
```bash
curl -sf http://192.168.53.251:8123/ && echo "OK, toi duoc ClickHouse" || echo "KHONG toi duoc"
```
Nếu không tới được, cần đổi `CLICKHOUSE_HOST` sang 1 ClickHouse mà server này thực sự với tới
(xem mục [3. Kết nối DATA](#3-kết-nối-data-clickhouse) bên dưới) hoặc setup VPN/tunnel trước.

### 6. Đổi `API_KEY` trong `.env`

File `.env` đã có sẵn 1 key được tạo riêng cho bản clone này (khác key trên máy Windows) —
đổi lại nếu muốn:
```bash
nano .env   # sửa dòng API_KEY=...
```

### 7. Chạy

```bash
./start-observability-agent.sh
```
Hoặc chạy nền lâu dài (khuyên dùng cho server) bằng `systemd` hoặc `tmux`/`screen` thay vì để
terminal mở — xem mục [Chạy nền trên server](#chạy-nền-trên-server-tmux--systemd) bên dưới.

---

## Kiến trúc tổng quan

```
                        ┌─────────────────────────┐
   Trình duyệt (bạn) ──▶│  static/index.html      │
   http://127.0.0.1:8787│  (chat đơn giản)        │
                        └───────────┬─────────────┘
                                    │ fetch() + X-API-Key
                                    ▼
                        ┌─────────────────────────┐
   LibreChat ──────────▶│  FastAPI (app/main.py)  │
   (custom endpoint,     │  - /api/chat            │
    Authorization:       │  - /api/alert           │
    Bearer <key>)        │  - /v1/chat/completions │◀── OpenAI-compatible,
                        │  - /v1/models           │    LibreChat gọi vào đây
                        └───────────┬─────────────┘
                                    │
                                    ▼
                        ┌─────────────────────────┐
                        │  Claude Agent SDK        │
                        │  (app/agent.py)          │
                        │  query() / stream_agent()│
                        └───────────┬─────────────┘
                            │               │
                 auth: Claude Code    tool: MCP ClickHouse
                 login (Pro/Max)      (uvx mcp-clickhouse)
                            │               │
                            ▼               ▼
                  claude.exe bundled   ClickHouse (host:port
                  trong claude-agent-  trong app/config.py)
                  sdk package
```

Có 2 lớp entry hoàn toàn độc lập vào cùng 1 "bộ não" (`app/agent.py`):

| Entry | File | Dùng khi |
|---|---|---|
| Web chat riêng | `static/index.html` → `/api/chat`, `/api/alert` | Tự test nhanh, không cần mở LibreChat |
| LibreChat | `/v1/chat/completions`, `/v1/models` | Muốn dùng chung giao diện LibreChat, có sẵn model khác (Ollama...) để so sánh |

---

## 1. Kết nối MODEL (Claude) — auth hoạt động thế nào

**Không dùng `ANTHROPIC_API_KEY`.** File `app/agent.py` gọi thẳng `claude_agent_sdk.query()`,
SDK này tự động dùng **credential Claude Code đã đăng nhập sẵn trên máy** (qua `ant auth login`
hoặc login trực tiếp trong Claude Code). Cụ thể:

- Package `claude-agent-sdk` (cài trong `.venv`) đóng gói sẵn 1 bản binary Claude Code thật
  (trên Linux: `.venv/lib/python3.13/site-packages/claude_agent_sdk/_bundled/claude`, không có
  đuôi `.exe`).
- Khi `query()`/`ClaudeSDKClient` chạy, nó spawn subprocess binary này, và nó tự đọc credential
  đã lưu từ lần `claude login` trước **trên chính server này** (xem mục "Cài đặt" bước 4 — đây
  là credential riêng của server, không dùng chung được với máy Windows).
- **Không cần set biến môi trường gì cho việc auth cả** — nếu Claude Code trên máy đang login
  bình thường thì agent này dùng ngay được, miễn phí theo quota Pro/Max, không tính thêm
  token như API riêng.
- Cấu hình model/thinking nằm trong `app/agent.py` hàm `_build_options()`:
  - `thinking={"type": "adaptive", "display": "summarized"}` — bật suy nghĩ, hiển thị tóm tắt
    (để stream ra được phần "đang suy nghĩ" cho người dùng thấy).
  - `permission_mode="default"` + `allowed_tools=["mcp__clickhouse__*"]` — tự động cho phép
    gọi tool ClickHouse, không cần approval thủ công.

**Nếu muốn đổi sang API key thật** (trả phí, không giới hạn quota, dùng được cho nhiều người):
xóa hẳn cách tiếp cận Agent SDK này, quay lại dùng SDK `anthropic` thường (`pip install anthropic`)
với `ANTHROPIC_API_KEY` — khác hoàn toàn code hiện tại, không phải chỉnh 1 dòng.

### Rủi ro điều khoản sử dụng

Docs chính thức của Claude Agent SDK ghi rõ: không được dùng quota claude.ai/Pro cho sản phẩm
phục vụ **nhiều người khác** (teammate, automation dùng chung) trừ khi Anthropic duyệt trước.
Anh đã quyết định **chấp nhận rủi ro này ở mức tự dùng cá nhân**, có mở rộng thêm sang LibreChat
(vẫn trên máy này, vẫn 1 mình dùng) — nhưng **không** nên:
- Mở port ra ngoài LAN cho người khác truy cập
- Gắn `/api/alert` vào workflow n8n chạy tự động phục vụ cả team
- Chia sẻ API key (`.env` → `API_KEY`) cho người khác dùng

---

## 2. Kết nối ACCOUNT — API key nội bộ (không phải Claude)

Đừng nhầm 2 khái niệm "key" khác nhau trong hệ thống này:

| Key | Nằm ở đâu | Dùng để làm gì |
|---|---|---|
| **Claude credential** | Do Claude Code quản lý trên server này (không nằm trong project) | Auth thật với Anthropic — xem mục "Cài đặt" bước 4 |
| **`API_KEY` trong `.env`** | `~/observability-agent/.env` | Khóa tự đặt, chỉ để **chặn người khác gọi vào server FastAPI của anh** (không liên quan gì Claude) |

Cách dùng `API_KEY` nội bộ:
- Web chat (`static/index.html`): nhập vào ô "API Key" trên giao diện → gửi header `X-API-Key`.
- LibreChat: khai trong `librechat.yaml` → `endpoints.custom` → mục `Observability Agent` →
  field `apiKey` (LibreChat tự gửi thành header `Authorization: Bearer <apiKey>`).
- Muốn đổi key: sửa `.env` → `API_KEY=...` → sửa luôn `apiKey` trong `librechat.yaml` cho khớp
  → restart cả 2 server (agent + LibreChat backend).

---

## 3. Kết nối DATA (ClickHouse)

Nằm 100% trong `app/config.py`, biến `CLICKHOUSE_MCP`:

```python
CLICKHOUSE_MCP = {
    "clickhouse": {
        "command": "uvx",
        "args": ["--with", "mcp-clickhouse", "--python", "3.10", "mcp-clickhouse"],
        "env": {
            "CLICKHOUSE_HOST": "192.168.53.251",   # ← đổi host ở đây
            "CLICKHOUSE_PORT": "8123",
            "CLICKHOUSE_USER": "default",
            "CLICKHOUSE_PASSWORD": "",              # rỗng = không auth (UAT hiện tại)
            "CLICKHOUSE_SECURE": "false",
        },
    }
}
```

- Đây là **cùng 1 MCP server** (`mcp-clickhouse`, chạy qua `uvx`, không cài gì thêm) mà
  LibreChat cũng dùng — copy y hệt config từ `librechat.yaml` sang đây.
- Hiện đang trỏ vào **ClickHouse UAT** (`192.168.53.251:8123`, user `default`, không password) —
  đây **không phải** cluster production `hsc_log_mgmt`. Agent tự nhận ra và nói rõ điều này khi
  trả lời (vì nó còn nhớ context từ Claude Code trên máy — xem mục "Lưu ý" bên dưới).
- **Muốn trỏ sang production**: sửa `CLICKHOUSE_HOST`/`PORT`/`USER`/`PASSWORD` cho đúng, restart
  server. ⚠️ Nhớ nguyên tắc cũ: nếu `PASSWORD` để rỗng thì **hardcode `""`** trực tiếp, không dùng
  `${VAR}` kiểu LibreChat — vì đây là code Python thuần, biến môi trường phải tự đọc qua `.env`
  và gán tay vào dict này (không có cơ chế `${}` tự thay thế như LibreChat yaml).
- Tool được phép gọi: `mcp__clickhouse__list_databases`, `mcp__clickhouse__list_tables`,
  `mcp__clickhouse__run_query` (đúng 3 tool có sẵn trong `mcp-clickhouse`, không tự thêm được
  tool khác trừ khi sửa MCP server).

---

## Cấu trúc file

```
observability-agent/
├── .env                       # API_KEY nội bộ
├── app/
│   ├── config.py               # MCP ClickHouse config (mục 3) + 2 system prompt
│   ├── agent.py                # gọi Claude Agent SDK, streaming, thinking, tool status
│   ├── main.py                 # FastAPI: /api/chat, /api/alert, serve static/
│   └── openai_compat.py        # /v1/chat/completions, /v1/models cho LibreChat
├── static/
│   └── index.html              # trang chat riêng, tự test không cần LibreChat
├── smoke_test.py                # script test độc lập, không qua HTTP — chạy khi nghi ngờ lỗi auth/MCP
├── start-observability-agent.sh   # cách chạy trực tiếp (không Docker), 127.0.0.1:8787
├── Dockerfile                     # cách chạy bằng Docker (mục riêng bên dưới)
├── docker-compose.yml              # kèm volume persist login + port binding local-only
└── .dockerignore
```

## Cách chạy

```bash
./start-observability-agent.sh
```
`Ctrl+C` để tắt. Chỉ nghe `127.0.0.1` (không ai ngoài chính server này gọi vào được, kể cả trong
cùng mạng nội bộ — muốn LibreChat trên máy khác gọi vào phải tự cân nhắc mở thêm, xem cảnh báo
rủi ro ở đầu file trước khi làm vậy).

- Web chat riêng: mở **http://127.0.0.1:8787/** (nếu server có GUI/browser; nếu headless thì
  SSH port-forward: `ssh -L 8787:127.0.0.1:8787 user@server` rồi mở trên máy local)
- LibreChat: nếu LibreChat cũng chạy trên **cùng server** này, khai `baseURL` trong
  `librechat.yaml` là `http://127.0.0.1:8787/v1/` giống hệt bản Windows. Nếu LibreChat chạy ở
  **máy/server khác**, đây chính là chỗ đụng vào giới hạn "chỉ 127.0.0.1" — cần cân nhắc kỹ
  trước khi đổi sang bind IP khác (xem [Rủi ro điều khoản sử dụng](#rủi-ro-điều-khoản-sử-dụng)).

## Chạy bằng Docker (thay thế cho cách cài trực tiếp ở trên)

> ⚠️ **Chưa build/test thật** — máy dùng để viết bản Docker này không cài Docker, nên
> `Dockerfile`/`docker-compose.yml` dưới đây được viết cẩn thận theo đúng kiến thức Docker/best
> practice nhưng **chưa tự `docker build` kiểm chứng**. Anh build lần đầu nhớ theo dõi kỹ log,
> báo em nếu lỗi để sửa.

### Vì sao cần chú ý riêng phần login

Auth vẫn đi theo đúng cơ chế cũ: **không dùng `ANTHROPIC_API_KEY`**, mà dựa vào credential
`claude login` lưu trên đĩa. Vấn đề là container Docker mặc định **không có gì tồn tại sau khi
xóa** — nếu không xử lý riêng, mỗi lần build/chạy lại container sẽ mất login, phải đăng nhập lại
liên tục. Cách xử lý trong `docker-compose.yml`:

- **Named volume `claude-home` mount vào `/home/agent`** (toàn bộ thư mục home của user chạy
  app trong container) — vì file/thư mục chính xác Claude Code lưu credential ở đâu dưới `$HOME`
  không phải thứ em nắm chắc 100%, mount nguyên cả `$HOME` để không bỏ sót file nào, credential
  sẽ **sống sót qua các lần** `docker compose down` / `docker compose up` / rebuild image (chỉ
  mất khi xóa hẳn volume bằng `docker compose down -v` hoặc `docker volume rm`).
- **Login `claude` là bước tách riêng, chạy 1 lần, bằng 1 image khác hẳn (`node:22-slim`)**,
  không nằm trong image chính (`ai-agent`) — vì server chạy headless nên không thể tự động hoá
  bước đăng nhập tương tác, và vì `claude` CLI chỉ cần dùng đúng 1 lần này nên tách riêng cho
  gọn. 2 image dùng chung volume `claude-home` nên credential login ở image tạm vẫn được image
  chính đọc thấy. (Node vẫn có trong image chính — không phải để login, mà vì `claude` CLI cài
  ở bước login là 1 script Node mà `claude_agent_sdk` gọi tới ở mọi request, xem mục
  [claude-mem](#claude-mem--đã-bỏ-hẳn-dùng-clickhouse-memory-thay-thế) bên dưới.)

### Các bước

**1. Build image chính:**
```bash
docker compose build ai-agent
```

**2. Login lần đầu (bắt buộc, chạy tương tác qua chính terminal SSH đang mở):**
```bash
docker compose run --rm login
```
Lệnh này tải `node:22-slim` (nếu chưa có), cài `claude` CLI ngay lúc chạy (không build sẵn vào
image), rồi mở `claude login`. Làm theo hướng dẫn trên màn hình — nếu nó cố mở trình duyệt (sẽ
fail vì container không có GUI/browser), tìm cờ kiểu `--no-browser` (chạy
`docker compose run --rm login sh -c "claude login --help"` xem các cờ hỗ trợ) để lấy URL dán
sang máy có trình duyệt, đăng nhập, rồi paste code xác nhận ngược lại terminal — **giống hệt
flow đã làm ở bản cài trực tiếp**, chỉ khác là chạy trong container. Container `login` tự xóa
sau khi thoát (`--rm`), nhưng credential đã ghi vào volume `claude-home` thì vẫn còn.

**3. Kiểm tra login thành công** (chạy `smoke_test.py` trong container tạm, dùng chung volume):
```bash
docker compose run --rm ai-agent uv run smoke_test.py
```
Ra dữ liệu ClickHouse thật → login OK.

**4. Chạy service thật (nền, tự khởi động lại nếu crash):**
```bash
docker compose up -d
docker compose logs -f     # xem log
```

**5. Kiểm tra:**
```bash
curl -H "Authorization: Bearer $(grep API_KEY .env | cut -d= -f2)" http://127.0.0.1:8787/v1/models
```

### claude-mem — đã bỏ hẳn, dùng ClickHouse memory thay thế

Trước đây có thử tích hợp `claude-mem` (memory chung của Claude Code, lưu ở `$HOME/.claude-mem/`)
để agent nhớ thông tin giữa các phiên chat. **Đã bỏ hoàn toàn**, thay bằng bảng
`hsc.agent_memory_dist` trong ClickHouse (xem `MEMORY_SETUP.md` + mục "Giới hạn quyền của agent"
bên dưới) — lý do: `claude-mem` cần `setting_sources=["user"]` để agent đọc được hook/MCP nó
đăng ký, nhưng dòng đó kéo theo **toàn bộ** `~/.claude/settings.json` + hook cấp máy (bash
pre-approved của các dự án khác, v.v.) vào agent — trái với mục tiêu giới hạn agent chỉ trong
phạm vi observability/ClickHouse. Đã gỡ `RUN npm install -g claude-mem` khỏi `Dockerfile`,
`setting_sources` khỏi `app/agent.py`, và `mcp__claude-mem__*` khỏi `ALLOWED_TOOLS`. Node vẫn
còn trong image chính — **không phải** cho claude-mem, mà vì `@anthropic-ai/claude-code` (cài
qua service `login`, xem bước 2) là 1 CLI chạy bằng Node mà `claude_agent_sdk` gọi tới ở mọi
request, nên vẫn cần `node` binary sẵn trong image chính dù không `npm install` gì thêm ở đó.

### Giới hạn quyền của agent (chỉ phục vụ observability)

`app/config.py` khai báo rõ các tầng chặn để agent **không** thực thi lệnh hệ thống, không tự
đọc/ghi file trên server, không gọi ra ngoài phạm vi observability đã duyệt:

- `ALLOWED_TOOLS` — whitelist tường minh: `mcp__clickhouse__*` (đọc log/metrics), 
  `mcp__clickhouse-memory__*` (đọc/ghi `hsc.agent_memory_dist`, đã giới hạn ở tầng ClickHouse
  user — xem `MEMORY_SETUP.md`), và từng tool ĐỌC của Grafana liệt kê tên rõ ràng (không dùng
  `mcp__grafana__*` — wildcard sẽ vô tình mở luôn `update_dashboard`/`patch_dashboard`/
  `alerting_manage_*`/`create_incident`/tool admin).
- `DISALLOWED_TOOLS` — chặn cứng toàn bộ tool có sẵn của Claude Code (`Bash`, `Read`, `Write`,
  `Edit`, `Glob`, `Grep`, `WebSearch`, `WebFetch`, `Task`, `SlashCommand`, ...) ở tầng SDK, không
  phụ thuộc allowed_tools hay permission_mode. Agent **không có** khả năng tự mở file trên server.
- Không set `setting_sources` → agent không đọc `~/.claude/settings.json`/hooks/skills cấp máy.

Muốn kiểm tra lại đúng chưa: gọi `/api/chat` bảo agent "chạy lệnh `ls`" hoặc "đọc file X trên
server" — phải bị từ chối/không có tool nào thực hiện được, chỉ trả lời bằng text.

**Đọc ảnh/file — không qua tool `Read`, mà qua đính kèm trực tiếp trong request.** Vì agent
không có tool filesystem, cách duy nhất để nó "thấy" ảnh/file là caller tự đọc, encode base64,
và gửi kèm trong `attachments` của `/api/chat`:
```bash
curl -X POST http://127.0.0.1:8787/api/chat \
  -H "X-API-Key: $(grep API_KEY .env | cut -d= -f2)" \
  -H "Content-Type: application/json" \
  -d "{
    \"session_id\": \"t1\",
    \"message\": \"Phân tích ảnh dashboard này giúp anh\",
    \"attachments\": [{
      \"media_type\": \"image/png\",
      \"data\": \"$(base64 -w0 screenshot.png)\",
      \"filename\": \"screenshot.png\"
    }]
  }"
```
- `image/*` và `application/pdf` → gửi thẳng dạng content block chuẩn Messages API (model đọc
  trực tiếp, không qua OCR/parse thủ công).
- Loại file khác (`.txt`, `.csv`, `.log`...) → server tự decode base64 thành text và chèn vào
  prompt kèm tên file, vì Claude không có block "document" riêng cho các định dạng này.
- Endpoint `/api/alert` chưa hỗ trợ `attachments` (chỉ nhận payload JSON của alert) — nếu cần,
  báo lại, thêm tương tự không khó.

### Grafana MCP

`GRAFANA_URL` + `GRAFANA_SERVICE_ACCOUNT_TOKEN` khai báo trong `.env` (để trống thì service vẫn
chạy, chỉ tool Grafana lỗi khi agent gọi tới). Tạo service account trong Grafana với role
**Viewer** là đủ — khớp với việc `ALLOWED_TOOLS` chỉ mở tool đọc (dashboard, Prometheus, Loki,
incident, on-call), không mở tool ghi/admin. Server chạy qua `uvx mcp-grafana` (package chính
thức `grafana/mcp-grafana`), giống hệt pattern `uvx mcp-clickhouse` đã dùng — không cần cài gì
thêm vào `Dockerfile`.

### Cập nhật code sau này

```bash
docker compose down
docker compose build
docker compose up -d
```
Volume `claude-home` không bị đụng tới khi làm vậy — login vẫn còn nguyên, không cần đăng nhập
lại. Chỉ mất login nếu chủ động xóa volume.

## 4. Antigravity (nghiên cứu / deepfake detection)

Agent giờ đã có scaffold để tích hợp Antigravity (mục đích nghiên cứu, ví dụ luận văn phát
hiện deepfake). Tính năng này là tùy chọn — không được bật mặc định. Những bước cơ bản:

1. Cài thêm SDK Antigravity (tùy chọn). Có hai cách:

```bash
# (a) uv extras nếu dùng uv/uvx to manage deps
uv sync --extras antigravity

# (b) hoặc pip install trực tiếp từ GitHub
pip install "git+https://github.com/google-antigravity/antigravity-sdk-python.git"
```

2. (Tùy) cấu hình MCP nghiên cứu cho Antigravity

File `app/config.py` chứa một placeholder `ANTIGRAVITY_MCP` (key: `antigravity-research`).
Nếu bạn có một MCP implementation `mcp-antigravity` (ví dụ wrapper giúp model truy cập dữ
liệu nội bộ, hoặc expose đánh giá deepfake qua tool uvx), điền biến môi trường sau vào
`.env` và cấu hình MCP tương ứng:

- `ANTIGRAVITY_ENDPOINT` — URL endpoint nếu MCP cần gọi service external.
- `ANTIGRAVITY_API_KEY` — nếu cần key để gọi service.

3. Gọi model Antigravity qua OpenAI-compat endpoint

Sau khi cài SDK, bạn có thể gọi model mới `antigravity-agent` qua endpoint `/v1/chat/completions`.
Ví dụ (một-thuần hoàn, non-streaming):

```bash
curl -s -X POST http://127.0.0.1:8787/v1/chat/completions \
  -H "Authorization: Bearer $(grep API_KEY .env | cut -d= -f2)" \
  -H "Content-Type: application/json" \
  -d '{"model":"antigravity-agent","messages":[{"role":"system","content":"You are a research assistant."},{"role":"user","content":"Phân tích khả năng đây là deepfake: <mô tả ảnh/video>"}]}'
```

Lưu ý:
- Hiện adapter `app/antigravity.py` là một scaffold: nếu SDK chưa cài, server trả lỗi 501.
- Tích hợp hiện thực hiện một-shot (không streaming). Nếu muốn streaming kết quả hoặc
  exposes tool via MCP, cần phát triển thêm `mcp-antigravity` và cập nhật adapter cho
  streaming/async theo API của SDK.
- Vì mục đích luận văn/research, hãy tách môi trường này ra (container/VM riêng) để tránh
  lẫn lộn credential/quota với hệ thống production.

Nếu bạn muốn, tôi có thể:
- tạo ví dụ `mcp-antigravity` stub (uvx tool) để test local,
- hoặc bổ sung streaming support nếu Antigravity SDK hỗ trợ streaming.

## Chạy nền trên server (tmux / systemd)

**Cách nhanh (tmux)** — phù hợp để test:
```bash
tmux new -s observability-agent
./start-observability-agent.sh
# Ctrl+B rồi D để detach, tmux attach -t observability-agent để quay lại
```

**Cách bền vững (systemd)** — phù hợp chạy lâu dài, tự khởi động lại nếu crash:
```ini
# /etc/systemd/system/observability-agent.service
[Unit]
Description=Observability Agent
After=network.target

[Service]
Type=simple
User=<user chạy, không nên là root>
WorkingDirectory=/home/<user>/observability-agent
ExecStart=/home/<user>/observability-agent/start-observability-agent.sh
Restart=on-failure

[Install]
WantedBy=multi-user.target
```
```bash
sudo systemctl daemon-reload
sudo systemctl enable --now observability-agent
sudo journalctl -u observability-agent -f   # xem log
```

## Hiệu năng — vì sao chậm và đã fix thế nào

**Lượt chat đầu tiên của 1 hội thoại luôn chậm (~15-18s)** — đó là chi phí spawn subprocess
`claude.exe` (+ MCP ClickHouse), không tránh được, đo trực tiếp bằng cách tách riêng cho thấy
việc "khởi động" chiếm phần lớn thời gian này.

**Fix đã áp dụng**: thay vì gọi `query()` (spawn tiến trình mới mỗi tin nhắn) như bản đầu,
`app/agent.py` giờ dùng `ClaudeSDKClient` — **kết nối 1 lần, giữ sống, tái sử dụng cho các lượt
sau trong cùng hội thoại** (`_client_pool` trong `agent.py`). Kết quả đo thực tế: lượt đầu ~15s,
lượt 2 trở đi ~8s (nhanh hơn 2 lần, không phải trả phí spawn lại).

- **`/api/chat`** (web chat riêng): key nhận diện hội thoại = `session_id` do client tự gửi —
  đơn giản, ổn định.
- **`/v1/chat/completions`** (LibreChat): LibreChat không gửi id hội thoại ổn định (chuẩn OpenAI
  không có field này), nên phải **fingerprint (băm) toàn bộ lịch sử tin nhắn** để nhận diện
  "đây có phải tiếp tục hội thoại cũ không". Nếu khớp → chỉ gửi tin nhắn mới, dùng lại client cũ.
  Nếu không khớp (hội thoại mới, hoặc server vừa restart mất client cũ) → tạo client mới, gửi lại
  toàn bộ transcript để khôi phục ngữ cảnh (chậm như lượt đầu).

**~8s còn lại mỗi lượt sau khi đã tối ưu là độ trễ model thật** (model xử lý + "suy nghĩ" theo
`thinking: adaptive`) — không phải lỗi code, không tối ưu thêm được nhiều trừ khi tắt bớt
`thinking` (đánh đổi mất phần hiển thị "đang suy nghĩ" đã làm ở phần trước).

⚠️ **Giới hạn đã biết**: client trong `_client_pool` không tự dọn dẹp — mỗi hội thoại còn "sống"
sẽ giữ 1 subprocess chạy mãi cho tới khi server restart. Với quy mô 1 người tự test thì không
vấn đề gì, nhưng nếu dùng lâu/nhiều hội thoại song song, RAM sẽ tăng dần (mỗi subprocess Claude
Code khá nặng). Restart server (`systemctl restart observability-agent` hoặc chạy lại
`start-observability-agent.sh`) định kỳ nếu thấy máy chậm/đầy RAM.

## Debug nhanh khi có lỗi

1. **Lỗi auth/MCP** → chạy `uv run smoke_test.py`, nếu ra dữ liệu ClickHouse thật là auth + MCP ổn.
2. **`CLINotFoundError` / lỗi spawn subprocess** → thường do 2 process cùng gọi MCP ClickHouse
   đồng thời (VD: agent LibreChat "Miku" vẫn còn gắn tool `clickhouse` riêng + gọi cả endpoint
   này cùng lúc) — bỏ tool MCP trùng ở phía kia, chỉ giữ 1 đường gọi.
3. **Port 8787 đã bị chiếm** → `sudo lsof -i :8787` hoặc `ss -tlnp | grep 8787` tìm PID, tắt
   process cũ (`kill <pid>`) trước khi chạy lại.
4. **"Failed to fetch" trên web chat** → kiểm tra thanh địa chỉ trình duyệt phải là
   `http://127.0.0.1:8787/`. Nếu truy cập từ máy khác qua SSH port-forward, đảm bảo đã
   `ssh -L 8787:127.0.0.1:8787 ...` đúng, không mở file `index.html` trực tiếp.
5. **`claude: command not found` hoặc auth lỗi ngay từ bước smoke_test** → chưa chạy `claude login`
   trên server này (bước 4 mục Cài đặt), hoặc login bị hết hạn — chạy lại `claude login`.

## Lưu ý quan trọng

Trên máy Windows dev, agent này **không phải Claude "trắng"** — vì dùng chung binary Claude Code
thật trên máy đó, nó kế thừa toàn bộ context/hook/memory hệ thống của Claude Code (ghi chú/
observation từ các phiên làm việc khác trên đúng máy đó, VD tự biết phân biệt ClickHouse UAT vs
production `hsc_log_mgmt`).

**Trên server Ubuntu này, hành vi sẽ KHÁC** — vì đây là 1 lần `claude login` hoàn toàn mới, server
này **không có** kho ghi chú/context tích lũy từ máy Windows. Agent sẽ trả lời "sạch" hơn, không
tự biết những thông tin ngữ cảnh đặc thù HSC ONE mà bản Windows biết sẵn — chỉ dựa vào
`system_prompt` trong `app/config.py` và dữ liệu thật truy vấn được qua ClickHouse MCP. Đây không
phải lỗi, chỉ là khác biệt cần biết trước khi so sánh kết quả 2 bản.
