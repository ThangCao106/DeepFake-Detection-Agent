# Deepfake Forensic Agent — SRE & Forensic Console
### Kiến trúc Giám sát An ninh Đa tầng & Giám định Bằng chứng Số (Dual-Engine: Claude SDK & Gemini SDK)

Hệ thống điều hành & giám định **Deepfake Forensic Agent** phục vụ kỹ sư **SRE, Đội ngũ An ninh Mạng (Red Team/Pentest)** và các nhà nghiên cứu:
- **Tra cứu chứng thư (ForensicCertificateVault)**: Xác minh chữ ký số SHA-256, ma trận bằng chứng 6 chiều (PRNU, 2D-FFT, Coherence) theo tiêu chuẩn EU AI Act Art. 50.
- **Hiệu chỉnh mô hình (Calibration)**: Theo dõi độ lệch tự tin qua Platt Scaling ($T^* = 1.6178$), tối ưu ngưỡng Youden's $J$ ($\tau^* = 0.8118$), giảm ECE $0.0894 \rightarrow 0.0195$ (-78.19%).
- **Quản lý & Thử nghiệm LoRA**: Giám sát Adapter Low-Rank Adaptation (r=16, $\alpha$=32, VRAM 7.2GB, cải thiện +8.4% độ chính xác cho ngữ cảnh truyền thông tiếng Việt).
- **Báo cáo An ninh & Tính toán**: Theo dõi tiến trình Colab Cloud GPU và kết quả Pentest theo chu kỳ **Daily (Hằng ngày)**, **Weekly (Hằng tuần)**, **Monthly (Hằng tháng)**.
- **Trợ lý AI Forensic Drawer**: Trực quan hóa tương tác streaming với Dual-Engine: **Claude 3.7 Sonnet** (Suy luận sâu) & **Google Gemini 2.5** (Tốc độ cao).
- **Cổng chuẩn OpenAI-Compatible** (`/v1/chat/completions`) tại cổng `127.0.0.1:8787`.

> ⚡ **Zero External DB Dependencies**: Hệ thống hoạt động độc lập, không cần ClickHouse, Grafana, n8n hay cơ sở dữ liệu bên ngoài. Khởi động tức thì trong 1 giây trên cả Linux (Ubuntu) và Windows.

---

## 🏗️ Sơ đồ Kiến trúc Hệ thống (Option B - Tiered SRE Pipeline)

```mermaid
graph TD
    User["👨‍💻 SRE Engineer / DevOps"] -->|Web UI: http://127.0.0.1:8787| WebUI["Web Control Panel (static/index.html)"]
    AlertSystem["🚨 Prometheus / Zabbix / Webhook"] -->|POST /api/alert| FastAPIGateway
    ThirdParty["🤖 Slack / Telegram / LibreChat"] -->|POST /v1/chat/completions| FastAPIGateway["FastAPI Gateway (app/main.py)"]

    WebUI --> FastAPIGateway

    subgraph "Core SRE Dual-Engine"
        FastAPIGateway --> Router["Intelligent SRE Router / Pool"]

        Router -->|"Tier 1: Triage nhanh, quét log, cảnh báo"| GeminiEngine["⚡ Google Gemini SDK (Tier 1)<br/>gemini-2.5-flash / gemini-2.5-pro"]
        Router -->|"Tier 2: RCA chuyên sâu, kiến trúc, phục hồi"| ClaudeEngine["🧠 Claude Agent SDK (Tier 2)<br/>claude-3-7-sonnet / claude-3-5-sonnet"]
    end

    subgraph "Safe SRE Tooling"
        ClaudeEngine --> Tools["Tool Sandbox (Chỉ đọc)<br/>- Read: Đọc file log<br/>- Grep: Tìm pattern lỗi<br/>- Glob: Quét danh mục log"]
    end
```

### Phân công nhiệm vụ giữa 2 Engine (Tiered Hybrid Architecture)
| Tầng | Model Engine | Nhiệm vụ chính trong SRE | Ưu thế vượt trội |
| :--- | :--- | :--- | :--- |
| **Tier 1 (Fast)** | **Gemini 2.5 Flash / Pro** | Phân loại alert tức thì, lọc log streaming dung lượng lớn, tóm tắt trạng thái uptime server. | Tốc độ phản hồi cực nhanh (< 1s), token window khổng lồ, chi phí tối ưu. |
| **Tier 2 (Deep)** | **Claude 3.7 Sonnet / 3.5** | Điều tra nguyên nhân gốc rễ (RCA) các lỗi phức tạp, phân tích race condition, sinh Runbook khắc phục. | Khả năng tư duy logic và lập luận code/hệ thống chuẩn xác nhất. |

---

## 🔑 Hướng dẫn Cấu hình Xác thực (Authentication Setup)

Hệ thống cho phép cấu hình linh hoạt cả **Claude** và **Gemini**:

### 1. Cấu hình Google Gemini SDK (Tier 1)
1. Lấy API Key miễn phí từ [Google AI Studio](https://aistudio.google.com/).
2. Cấu hình bằng **1 trong 2 cách**:
   * **Cách A (Trực quan trên Web UI)**: Mở `http://127.0.0.1:8787`, dán key `AIzaSy...` vào ô **"Gemini API Key"** rồi bấm **"Lưu Key Gemini"**. Hệ thống sẽ tự động cập nhật runtime và lưu vào file `.env`.
   * **Cách B (File cấu hình)**: Mở file `.env` và thêm:
     ```bash
     GEMINI_API_KEY=AIzaSy...
     ```

### 2. Cấu hình Claude Agent SDK (Tier 2)
Bạn có thể dùng **Claude API Key** trả phí hoặc **Tài khoản Claude Pro/Max (Claude Code CLI)**:
* **Cách A — Dùng Anthropic API Key (Khuyên dùng)**:
  Thêm vào file `.env` hoặc nhập trực tiếp trên thanh đăng nhập Web UI:
  ```bash
  CLAUDE_API_KEY=sk-ant-api03-...
  ```
* **Cách B — Dùng Claude Code CLI login (Tận dụng quota Claude Pro/Max)**:
  ```bash
  npm install -g @anthropic-ai/claude-code
  claude login
  ```
  *(Nếu chạy trên server không có GUI, CLI sẽ hiển thị URL xác thực OAuth PKCE, bạn mở link trên trình duyệt rồi dán mã code xác nhận ngược lại terminal hoặc Web UI)*.

---

## 🚀 Hướng dẫn Cài đặt & Vận hành

### Cách 1: Chạy trực tiếp với `uv` (Khuyên dùng — Cực nhẹ & Nhanh)

```bash
# 1. Tải uv nếu chưa có
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Cài đặt dependencies (tự động tải môi trường Python)
cd observability
uv sync

# 3. Kiểm tra kết nối nhanh (Smoke Test)
uv run smoke_test.py

# 4. Khởi chạy SRE Agent Server
uv run uvicorn app.main:app --host 127.0.0.1 --port 8787
```

### Cách 2: Chạy với Docker Compose

```bash
cd observability

# Khởi động dịch vụ nền
docker compose up -d

# Xem log hoạt động
docker compose logs -f
```

---

## 🧪 Kiểm tra Sức khỏe Hệ thống (Smoke Test)

Chạy script kiểm tra tự động trạng thái sẵn sàng của cả 2 Engine:
```bash
uv run smoke_test.py
```
**Kết quả hiển thị mẫu:**
```text
======================================================================
🚀 SRE OBSERVABILITY AGENT — DUAL-ENGINE SMOKE TEST
======================================================================

🔍 [1/2] Testing Google Gemini SDK (Tier 1 Engine)...
  🔑 Found GEMINI_API_KEY: AIzaSyD3...
  ✅ Gemini Response: PONG_SRE_OK: System telemetry operational...

🔍 [2/2] Testing Claude Agent SDK (Tier 2 Engine)...
  🛠️ SRE Whitelisted Tools: ['Read', 'Grep', 'Glob']
  ✅ Claude Response: CLAUDE_SRE_OK

======================================================================
📊 SMOKE TEST SUMMARY:
   - Google Gemini Engine (Tier 1): 🟢 ACTIVE
   - Claude Agent SDK Engine (Tier 2): 🟢 ACTIVE
======================================================================
```

---

## 📡 Danh mục API Phục vụ Giám sát & Tự động hóa

Mọi API bảo vệ bằng Header: `Authorization: Bearer <API_KEY>` (Mặc định trong `.env`: `observability-agent-key`).

### 1. OpenAI-Compatible Chat Completions (`POST /v1/chat/completions`)
Cung cấp khả năng chat tương thích chuẩn OpenAI, cho phép lựa chọn linh hoạt giữa các model:
* `sre-agent-hybrid`: Tự động điều phối kết hợp (Mặc định).
* `gemini-2.5-flash`: Triage tốc độ cao.
* `gemini-2.5-pro`: Phân tích log dung lượng lớn.
* `claude-3-7-sonnet`: Suy luận và điều tra nguyên nhân gốc rễ (RCA) chuyên sâu.

**Ví dụ gọi qua cURL:**
```bash
curl -X POST http://127.0.0.1:8787/v1/chat/completions \
  -H "Authorization: Bearer observability-agent-key" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemini-2.5-flash",
    "messages": [
      {"role": "user", "content": "Server bị lỗi HTTP 504 Gateway Timeout lúc 14:00, hãy phân tích các khả năng gây nghẽn."}
    ],
    "stream": false
  }'
```

### 2. Tự động hóa Xử lý Sự cố khẩn cấp (`POST /api/alert`)
Dành cho Webhook từ hệ thống giám sát (Alertmanager, Datadog, Zabbix):
```bash
curl -X POST http://127.0.0.1:8787/api/alert \
  -H "Authorization: Bearer observability-agent-key" \
  -H "Content-Type: application/json" \
  -d '{
    "service": "api-gateway",
    "status": "CRITICAL",
    "error_log": "Connection refused: database host 10.0.1.5:5432 unreachable after 30s timeout"
  }'
```
**JSON phản hồi có cấu trúc:**
```json
{
  "severity": "critical",
  "likely_root_cause": "Mất kết nối vật lý hoặc dịch vụ PostgreSQL tại 10.0.1.5 bị crash/quá tải kết nối (connection pool exhaustion).",
  "recommended_action": "Kiểm tra trạng thái service postgresql trên host 10.0.1.5, kiểm tra firewall port 5432 và xem log dmesg OOM-killer."
}
```

---

## 🛡️ Giới hạn & An toàn Bảo mật (Security Guardrails)
- **Safe Read-Only**: Agent **chỉ được phép đọc** file log qua `Read`, `Grep`, `Glob`.
- **Cấm thực thi tùy tiện**: Nghiêm cấm chạy lệnh shell (`Bash`, `KillShell`), cấm ghi/sửa file hệ điều hành (`Write`, `Edit`), cấm truy cập internet tùy tiện (`WebFetch`).
- **An toàn cục bộ**: Mặc định lắng nghe trên `127.0.0.1:8787`, xác thực chặt chẽ qua Bearer token.
