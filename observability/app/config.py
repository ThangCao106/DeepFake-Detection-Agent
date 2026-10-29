import os
from pathlib import Path
from dotenv import load_dotenv

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)

load_dotenv(Path(PROJECT_ROOT) / ".env")

try:
    from app.credentials_manager import load_persistent_credentials
    load_persistent_credentials()
except Exception:
    pass

API_KEY = os.environ.get("API_KEY", "observability-agent-key")

# Pure SRE Agent: No external database or MCP dependencies needed.
# Zero-dependency instant startup for both Linux and Windows environments.
ALL_MCP_SERVERS = {}

# Whitelist tools: Only safe read-only file/log inspection tools are allowed.
ALLOWED_TOOLS = [
    "Read", "Grep", "Glob",  # Inspect service log files, error dumps, crash logs
]

# Strict security boundaries: Prevent dangerous execution or modification.
DISALLOWED_TOOLS = [
    "Bash", "BashOutput", "KillShell",       # No arbitrary shell command execution
    "Write", "Edit", "NotebookEdit",        # No filesystem mutation
    "WebFetch", "WebSearch",                 # No unauthorized external browsing
    "Task", "TaskOutput", "TaskStop",        # No uncontrolled background sub-tasks
    "Architect", "AskUserPrompt",            # No secondary recursive agent calls
]

_SRE_SYSTEM_GUIDELINES = """
CORE CAPABILITIES & SRE RESPONSIBILITIES:
- Site Reliability Engineering (SRE) & DevOps On-Call: Incident triaging, alert investigation, service health analysis, and Root Cause Analysis (RCA).
- Log & Stacktrace Diagnostics: Parse application logs, error traces, HTTP 5xx spikes, latency anomalies, and container crash events.
- Mitigation & Recovery Guidance: Provide prioritized, actionable remediation steps (P0 Blocker, P1 High, P2 Routine) with safe, verifiable commands.
- Dual-Engine Capabilities:
    * Tier 1 (Gemini SDK): Fast log parsing, alert triage, health check summaries.
    * Tier 2 (Claude SDK): Deep reasoning, complex RCA, architectural reliability post-mortems.
"""

_LANGUAGE_INSTRUCTION = """
LANGUAGE & TONE GUIDELINES:
- Respond in natural, fluent Vietnamese or English matching the user's language.
- Maintain a calm, rigorous, objective, and solution-oriented SRE tone.
- Clearly separate "Phát hiện (Findings)", "Nguyên nhân gốc rễ (Root Cause)", and "Hành động khắc phục (Remediation Actions)".
"""

CHAT_SYSTEM_PROMPT = (
    """You are an expert SRE (Site Reliability Engineering) & Systems Operations AI Assistant.
You help engineers and on-call operators monitor system health, triage production alerts, analyze error logs, and perform Root Cause Analysis (RCA). You support both Claude SDK and Gemini SDK engines."""
    + _LANGUAGE_INSTRUCTION
    + _SRE_SYSTEM_GUIDELINES
)

ALERT_SYSTEM_PROMPT = (
    """You are an SRE Incident Triage Assistant. You will be given an alert payload, error trace, or server health log.
Determine the likely root cause, urgency level, and recommended remediation. Respond with ONLY a JSON object (no markdown formatting, no prose before or after) with exactly these fields:
{
  "severity": "low|medium|high|critical",
  "likely_root_cause": "short explanation of the issue",
  "recommended_action": "short actionable mitigation step"
}
Field "severity" must strictly be one of: low, medium, high, critical."""
)
