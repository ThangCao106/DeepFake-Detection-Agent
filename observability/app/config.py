import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)

load_dotenv(Path(PROJECT_ROOT) / ".env")

API_KEY = os.environ.get("API_KEY", "observability-agent-key")

# Read-only connection — Deepfake Detection Telemetry & System Analytics
DEEPFAKE_ANALYTICS_MCP = {
    "deepfake-analytics": {
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

# Persistent Read-Write Deepfake Agent Memory Table Store (deepfake_agent_memory)
# Scoped for benchmark runs, Colab training reports, and model weight history tracking.
DEEPFAKE_MEMORY_MCP = {
    "deepfake-agent-memory": {
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
ALL_MCP_SERVERS = {**DEEPFAKE_ANALYTICS_MCP, **DEEPFAKE_MEMORY_MCP, **GRAFANA_MCP, **ANTIGRAVITY_MCP}

# Whitelist — agent CHỈ được dùng đúng các tool liệt kê dưới đây, không hơn.
ALLOWED_TOOLS = [
    "mcp__deepfake-analytics__*",      # đọc log/metrics deepfake (read-only)
    "mcp__deepfake-agent-memory__*",   # đọc/ghi deepfake_agent_memory (read-write, scoped)
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
    "Globe", "Grep", "LS",                   # không tìm kiếm/liệt kê file
    "WebFetch", "WebSearch",                 # không truy cập internet
    "Task", "TaskOutput", "TaskStop",        # không tạo background task
    "Architect", "AskUserPrompt",            # không gọi agent con hay hỏi user
]

_DEEPFAKE_FORENSIC_INSTRUCTIONS = """
CORE CAPABILITIES & RESPONSIBILITIES:
- Multi-modal Deepfake Detection: Analyze and explain forensic indicators across Video (DF40, Celeb-DF v2, DFDC, DeeperForensics, Mega-MMDF), Image (CIFAKE, DFFD, iFakeFaceDB, Dima806), and Audio (ASVspoof 2021, DeepSpeak Audio, WaveFake, Omni-Fake).
- Calibration & Evaluation Metrics: Guide Expected Calibration Error (ECE), Maximum Calibration Error (MCE), Reliability Diagrams, and Temperature Scaling (z_i / T) optimization.
- Benchmark & Weight Tracking: Summarize daily benchmark reports (from report_colab/daily_benchmark/), Colab model training reports (from report_colab/YYYY-MM-DD/), and track model weight updates aggregated by date or weekly trends. Provide structured markdown tables and natural language summaries.
- Forensic Pipeline Diagnostics: Assist with log analysis, benchmark evaluation reports, confusion matrices, and confidence calibration.
"""

_LANGUAGE_INSTRUCTION = """
LANGUAGE & FORMATTING GUIDELINES:
- Respond in natural, fluent Vietnamese or English matching the user's query language. If the user asks in English, reply in clear, professional English. If the user asks in Vietnamese, reply in clear, natural Vietnamese without broken characters or garbled words.
- Format mathematical formulas clearly using LaTeX ($...$ for inline, $$...$$ for display equations) and ensure markdown tables are properly aligned.
- Maintain a scientific, precise, and helpful tone for deepfake detection research.
"""

CHAT_SYSTEM_PROMPT = (
    """You are an AI Deepfake Detection & Forensic Analysis Assistant for the DeepFake Detection Agent system.
You assist researchers and users with multi-modal deepfake detection (images, video, audio), benchmark evaluation (DeepfakeBench-MM, DF40, Omni-Fake, WaveFake), model calibration (ECE, Temperature Scaling), and detection pipeline diagnostics. You can also process uploaded media attachments (images, video frames, audio spectrographs, log files) when provided. Keep answers concise, scientific, and actionable."""
    + _LANGUAGE_INSTRUCTION
    + _DEEPFAKE_FORENSIC_INSTRUCTIONS
)

ALERT_SYSTEM_PROMPT = (
    """You are a Deepfake Detection Triage Assistant. You will be given a detection payload, anomaly report, or benchmark evaluation result.
Determine the likely root cause (e.g. low model confidence, spoof attack, out-of-distribution media, audio-visual mismatch) and recommended action. Respond with ONLY a JSON object (no markdown, no prose before or after) with exactly these fields:
{
  "severity": "low|medium|high|critical",
  "likely_root_cause": "short explanation",
  "recommended_action": "short actionable recommendation"
}
Field "severity" giữ nguyên giá trị tiếng Anh (low|medium|high|critical) đúng như trên — CHỈ nội dung mô tả trong "likely_root_cause" và "recommended_action" viết bằng tiếng Việt hoặc tiếng Anh phù hợp."""
    + _LANGUAGE_INSTRUCTION
    + _DEEPFAKE_FORENSIC_INSTRUCTIONS
)
