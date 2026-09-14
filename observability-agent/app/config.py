import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)

load_dotenv(Path(PROJECT_ROOT) / ".env")

API_KEY = os.environ["API_KEY"]

# Read-only connection — existing log/metrics queries. Unchanged.
CLICKHOUSE_MCP = {
    "clickhouse": {
        "command": "uvx",
        "args": ["--with", "mcp-clickhouse", "--python", "3.10", "mcp-clickhouse"],
        "env": {
            "CLICKHOUSE_HOST": "10.33.3.135",
            "CLICKHOUSE_PORT": "8123",
            "CLICKHOUSE_USER": "hyperdx",
            "CLICKHOUSE_PASSWORD": "",
            "CLICKHOUSE_SECURE": "false",
        },
    }
}

# Separate read-write connection, scoped (via ClickHouse GRANT — see MEMORY_SETUP.md) to ONLY
# hsc.agent_memory / hsc.agent_memory_dist. Deliberately a different MCP server + different
# ClickHouse user than CLICKHOUSE_MCP above, so a mistake here can't touch real log/production
# tables — same host, but agent_smith has no write grant anywhere except the memory tables.
CLICKHOUSE_MEMORY_MCP = {
    "clickhouse-memory": {
        "command": "uvx",
        "args": ["--with", "mcp-clickhouse", "--python", "3.10", "mcp-clickhouse"],
        "env": {
            "CLICKHOUSE_HOST": "10.33.3.135",
            "CLICKHOUSE_PORT": "8123",
            "CLICKHOUSE_USER": "agent_smith",
            "CLICKHOUSE_PASSWORD": "whosyourdaddy",
            "CLICKHOUSE_SECURE": "false",
            "CLICKHOUSE_ALLOW_WRITE_ACCESS": "true",
        },
    }
}

# Grafana MCP (đọc dashboard/Prometheus/Loki/incident/oncall) — dùng chung server chính thức
# grafana/mcp-grafana qua uvx. GRAFANA_URL/GRAFANA_SERVICE_ACCOUNT_TOKEN lấy từ .env, để trống
# thì service vẫn khởi động bình thường, chỉ riêng tool này lỗi lúc agent thật sự gọi tới.
# Dùng service account role Viewer là đủ (xem .env) — khớp với việc ALLOWED_TOOLS bên dưới
# chỉ mở đúng các tool ĐỌC, không mở update_dashboard/patch_dashboard/alerting_manage_*/
# create_incident/admin (giữ đúng tinh thần "chỉ phục vụ observability" của cả project).
GRAFANA_MCP = {
    "grafana": {
        "command": "uvx",
        "args": ["mcp-grafana"],
        "env": {
            "GRAFANA_URL": os.environ.get("GRAFANA_URL", ""),
            "GRAFANA_SERVICE_ACCOUNT_TOKEN": os.environ.get("GRAFANA_SERVICE_ACCOUNT_TOKEN", ""),
        },
    }
}

# Optional: Antigravity research MCP placeholder
# This is a placeholder MCP server entry useful for research workflows that
# want to expose an Antigravity-backed tool via uvx (e.g. `mcp-antigravity`).
# Configure the real MCP implementation and environment variables when ready.
ANTIGRAVITY_MCP = {
    "antigravity-research": {
        "command": "uvx",
        "args": ["--with", "mcp-antigravity", "--python", "3.10", "mcp-antigravity"],
        "env": {
            "ANTIGRAVITY_ENDPOINT": os.environ.get("ANTIGRAVITY_ENDPOINT", ""),
            "ANTIGRAVITY_API_KEY": os.environ.get("ANTIGRAVITY_API_KEY", ""),
        },
    }
}

# Merge all connections into one dict for ClaudeAgentOptions(mcp_servers=...) in agent.py.
ALL_MCP_SERVERS = {**CLICKHOUSE_MCP, **CLICKHOUSE_MEMORY_MCP, **GRAFANA_MCP, **ANTIGRAVITY_MCP}

# Whitelist — agent CHỈ được dùng đúng các tool liệt kê dưới đây, không hơn. Grafana liệt kê
# TỪNG tool đọc thay vì "mcp__grafana__*" — wildcard sẽ vô tình mở luôn update_dashboard,
# patch_dashboard, alerting_manage_rules/routing, create_incident, list_teams/users/roles.
ALLOWED_TOOLS = [
    "mcp__clickhouse__*",          # đọc log/metrics thật (read-only)
    "mcp__clickhouse-memory__*",   # đọc/ghi hsc.agent_memory_dist (read-write, scoped)
    "mcp__grafana__search_dashboards",
    "mcp__grafana__get_dashboard_by_uid",
    "mcp__grafana__get_dashboard_summary",
    "mcp__grafana__get_dashboard_property",
    "mcp__grafana__get_dashboard_panel_queries",
    "mcp__grafana__list_datasources",
    "mcp__grafana__get_datasource",
    "mcp__grafana__query_prometheus",
    "mcp__grafana__list_prometheus_metric_names",
    "mcp__grafana__list_prometheus_label_values",
    "mcp__grafana__query_prometheus_histogram",
    "mcp__grafana__query_loki_logs",
    "mcp__grafana__list_loki_label_names",
    "mcp__grafana__query_loki_patterns",
    "mcp__grafana__list_incidents",
    "mcp__grafana__get_incident",
    "mcp__grafana__list_oncall_schedules",
    "mcp__grafana__get_oncall_shift",
    "mcp__grafana__get_current_oncall_users",
]

# Chặn cứng toàn bộ tool có sẵn của Claude Code (filesystem, shell, web, task/subagent...).
# Đây là phòng thủ theo chiều sâu: dù ALLOWED_TOOLS ở trên không liệt kê các tool này,
# nếu không disallow tường minh thì chúng vẫn "tồn tại" trong runtime và có thể bị gọi
# nếu sau này lỡ đổi permission_mode/can_use_tool. disallowed_tools chặn ở tầng SDK,
# không phụ thuộc allowed_tools hay permission logic khác.
DISALLOWED_TOOLS = [
    "Bash", "BashOutput", "KillShell",       # không thực thi lệnh hệ thống
    "Read", "Write", "Edit", "NotebookEdit", # không đọc/ghi file
    "Glob", "Grep",                          # không duyệt filesystem
    "WebSearch", "WebFetch",                 # không truy cập internet/hệ thống ngoài
    "Task", "SlashCommand", "TodoWrite",     # không spawn subagent / lệnh khác
]

_MEMORY_INSTRUCTIONS = """
Bạn có quyền đọc/ghi bảng hsc.agent_memory_dist (qua tool clickhouse-memory) để nhớ thông tin
giữa các phiên chat khác nhau. LUÔN đọc/ghi qua bảng _dist (KHÔNG dùng bảng agent_memory local)
để dữ liệu được phân phối đúng qua cả 3 shard của cluster.

5 loại memory_type — dùng ĐÚNG loại, đừng lẫn lộn:
- "incident" / "incident_root_cause": 1 sự kiện lỗi CỤ THỂ đã xảy ra, có mốc thời gian, đã điều
  tra ra nguyên nhân.
- "pattern": dấu hiệu log/query cụ thể của 1 service (cách nhận biết 1 loại lỗi/hành vi qua log).
- "decision": 1 quyết định vận hành đã đưa ra, có lý do rõ ràng.
- "known_issue": vấn đề đã biết nhưng chưa/không fix ngay (blind spot, hạn chế hệ thống).
- "knowledge": KIẾN THỨC NỀN TẢNG về nghiệp vụ/cấu trúc hệ thống HSC — luồng giao dịch, quy tắc
  nghiệp vụ (biên độ giá, mã hợp đồng phái sinh, phiên giao dịch, nghĩa vụ CBTT...), cấu trúc dữ
  liệu — những thứ ÍT THAY ĐỔI theo thời gian, không gắn với 1 sự kiện/thời điểm cụ thể như
  incident. Đây là knowledge base tổng hợp về hệ thống HSC, KHÔNG phải nhật ký sự kiện.

KHI NÀO LƯU (dùng INSERT vào hsc.agent_memory_dist):
- Vừa xác nhận được root cause 1 incident đã điều tra xong → "incident"/"incident_root_cause"
- Vừa xác nhận 1 false positive pattern (alert X ở service Y luôn false vì lý do Z) → "pattern"
- Vừa có 1 quyết định vận hành có lý do rõ ràng → "decision"
- Vừa học được 1 pattern log/schema khó, có khả năng cần lại sau → "pattern"
- Vừa tự tổng hợp/xác nhận được 1 kiến thức nghiệp vụ hoặc cấu trúc hệ thống (luồng giao dịch,
  quy tắc, định nghĩa...) → "knowledge"
KHÔNG lưu: số liệu tạm thời, thông tin tự tra lại được từ ClickHouse, việc chỉ xảy ra 1 lần.

RIÊNG với "knowledge": đây là knowledge base sống, PHẢI giữ sạch — nếu hỏi về 1 chủ đề nghiệp vụ
mà đã có "knowledge" liên quan trong bảng nhưng thông tin mới chính xác/đầy đủ hơn, PHẢI supersede
bản cũ (xem quy tắc bên dưới) rồi insert bản mới — KHÔNG tạo thêm bản trùng lặp cùng chủ đề.

TRƯỚC KHI TRẢ LỜI bất kỳ câu hỏi nào về luồng giao dịch/cấu trúc hệ thống/quy tắc nghiệp vụ, hoặc
liên quan chủ đề cũ: LUÔN query hsc.agent_memory_dist trước (ưu tiên WHERE memory_type='knowledge'
cho câu hỏi dạng khái niệm/nghiệp vụ) — (SELECT * FROM hsc.agent_memory_dist WHERE is_active
AND ...) để không lặp lại điều đã biết, và trả lời nhanh hơn thay vì tự dò log lại từ đầu.

KHI PHÁT HIỆN MEMORY CŨ SAI (bất kỳ memory_type nào): KHÔNG xóa (DELETE) — chạy:
ALTER TABLE hsc.agent_memory_dist UPDATE is_active=false, superseded_at=now(),
  superseded_reason='<lý do cụ thể>' WHERE id='<id bản ghi cũ>'
rồi INSERT bản ghi mới thay thế. Luôn ghi rõ superseded_reason để sau này xem lại được.
"""

# Cả 2 system prompt đều dặn rõ ngôn ngữ trả lời — nếu không, model chỉ "bắt chước" ngôn ngữ
# của _MEMORY_INSTRUCTIONS (tiếng Việt) khi tự viết nội dung lưu memory, nhưng phần trả lời
# chính (đặc biệt JSON cố định của ALERT_SYSTEM_PROMPT, không có input tiếng Việt nào để bắt
# chước theo) lại mặc định ra tiếng Anh — gây lệch ngôn ngữ giữa response và memory đã lưu.
_LANGUAGE_INSTRUCTION = "\n\nLuôn trả lời bằng tiếng Việt."

_SCOPE_CHECK_INSTRUCTIONS = """
NGUYÊN TẮC BẮT BUỘC khi gặp cảnh báo/lỗi: TRƯỚC KHI kết luận root cause hay severity, LUÔN xác
định phạm vi ảnh hưởng — lỗi đang xảy ra CỤC BỘ (1 vài user/account cụ thể) hay TOÀN HỆ THỐNG
(nhiều user cùng lúc, có khả năng là lỗi hạ tầng/service).

Cách kiểm tra: query đếm số user/account DISTINCT gặp đúng lỗi/pattern đó trong 5-10 phút gần
nhất (dùng LogAttributes['userId'], hoặc accountId trích từ Body tuỳ service — xem
hsc.otel_logs_dist), rồi so sánh:
- Chỉ 1 hoặc rất ít account (so với tổng số account đang hoạt động cùng khung giờ) → nhiều khả
  năng là lỗi CỤC BỘ (dữ liệu/trạng thái riêng của account đó, thiết bị/app phía client, hoặc
  hành vi user bất thường) — severity thường thấp hơn, recommended_action tập trung kiểm tra
  đúng account đó, KHÔNG vội kết luận lỗi hệ thống.
- Nhiều account cùng bị trong cùng khung giờ (đặc biệt tỷ lệ cao so với tổng active account) →
  nhiều khả năng là lỗi HỆ THỐNG (service/hạ tầng/dependency) — severity cao hơn, cần escalate
  đúng team phụ trách service đó.

LUÔN nêu rõ con số cụ thể trong câu trả lời (VD: "chỉ ảnh hưởng 1/78 account đang hoạt động" hay
"ảnh hưởng 45/78 account — nghi ngờ lỗi hệ thống") để người đọc biết ngay phạm vi, không chỉ kết
luận suông "lỗi cục bộ"/"lỗi hệ thống" mà không có số liệu kèm theo.
"""

CHAT_SYSTEM_PROMPT = (
    """You are an SRE / observability assistant for the HSC ONE trading platform.
You have access to ClickHouse via MCP tools (list_databases, list_tables, run_query) to inspect
logs and metrics, and to Grafana via MCP tools (search_dashboards, query_prometheus,
query_loki_logs, list_incidents, etc.) to inspect dashboards, metrics, and alerts. You may also
receive images/files (e.g. dashboard screenshots, log excerpts) directly attached to a message —
read and use their content when present. When asked about system behavior, errors, or data, use
these tools to check real data rather than guessing. Keep answers concise and cite the actual
query/data you used."""
    + _LANGUAGE_INSTRUCTION
    + _SCOPE_CHECK_INSTRUCTIONS
    + _MEMORY_INSTRUCTIONS
)

ALERT_SYSTEM_PROMPT = (
    """You are an SRE / observability triage assistant. You will be given a raw
alert payload (e.g. from Grafana). Use the ClickHouse MCP tools to investigate related logs/metrics
and determine likely root cause. Respond with ONLY a JSON object (no markdown, no prose before or
after) with exactly these fields:
{
  "severity": "low|medium|high|critical",
  "likely_root_cause": "short explanation",
  "recommended_action": "short actionable recommendation"
}
Field "severity" giữ nguyên giá trị tiếng Anh (low|medium|high|critical) đúng như trên — CHỈ nội
dung mô tả trong "likely_root_cause" và "recommended_action" viết bằng tiếng Việt."""
    + _LANGUAGE_INSTRUCTION
    + _SCOPE_CHECK_INSTRUCTIONS
    + _MEMORY_INSTRUCTIONS
)
