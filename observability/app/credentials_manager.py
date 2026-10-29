"""
Persistent Multi-Layer Credentials Manager for Deepfake Forensic Agent.

Ensures API keys (Gemini, Claude, Meta Muse) persist permanently across:
  1. Container restarts (docker compose restart)
  2. Container recreations (docker compose up -d --force-recreate)
  3. Git CI/CD deployments and redeployments

Persistence hierarchy:
  - Level 1: Persistent Docker Named Volume: /home/agent/.credentials/forensic_env.json
  - Level 2: Bind-mounted host storage: /app/report_colab/.credentials/forensic_env.json & .env.backup
  - Level 3: App environment file: /app/.env or project root .env
"""
import os
import json
from pathlib import Path
from typing import Dict, Any, Optional

def _get_candidate_paths():
    base = Path(__file__).resolve().parent.parent
    return [
        Path("/home/agent/.credentials/forensic_env.json"),
        Path("/app/report_colab/.credentials/forensic_env.json"),
        base.parent / "report_colab" / ".credentials" / "forensic_env.json",
        base / ".env",
        base.parent / ".env",
    ]

def load_persistent_credentials() -> Dict[str, str]:
    """Load credentials from persistent volume or bind-mount storage into os.environ."""
    loaded = {}
    for p in _get_candidate_paths():
        try:
            if p.suffix == ".json" and p.exists():
                data = json.loads(p.read_text(encoding="utf-8"))
                for k, v in data.items():
                    val = str(v).strip()
                    if val and not os.environ.get(k):
                        os.environ[k] = val
                        loaded[k] = val
            elif p.name == ".env" and p.exists():
                for line in p.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        k, v = k.strip(), v.strip().strip('"\'')
                        if v and not os.environ.get(k):
                            os.environ[k] = v
                            loaded[k] = v
        except Exception:
            continue
    return loaded

def save_persistent_credentials(updates: Dict[str, Any]) -> None:
    """Save updated credentials to all persistent storage locations and .env."""
    if not updates:
        return

    # Update current process memory
    for k, v in updates.items():
        if v is not None:
            os.environ[k] = str(v).strip()

    # 1. Update JSON files across all persistent targets
    for p in _get_candidate_paths():
        if p.suffix == ".json":
            try:
                p.parent.mkdir(parents=True, exist_ok=True)
                current = {}
                if p.exists():
                    try:
                        current = json.loads(p.read_text(encoding="utf-8"))
                    except Exception:
                        current = {}
                for k, v in updates.items():
                    if v is not None:
                        current[k] = str(v).strip()
                p.write_text(json.dumps(current, indent=2, ensure_ascii=False), encoding="utf-8")
            except Exception:
                continue

    # 2. Update .env.backup shell file in report_colab (bind-mounted directly to host)
    backup_candidates = [
        Path("/app/report_colab/.credentials/.env.backup"),
        Path(__file__).resolve().parent.parent.parent / "report_colab" / ".credentials" / ".env.backup"
    ]
    for b_path in backup_candidates:
        try:
            b_path.parent.mkdir(parents=True, exist_ok=True)
            existing_lines = b_path.read_text(encoding="utf-8").splitlines() if b_path.exists() else []
            env_map = {}
            for line in existing_lines:
                if "=" in line and not line.strip().startswith("#"):
                    k, v = line.split("=", 1)
                    env_map[k.strip()] = v.strip()
            for k, v in updates.items():
                if v is not None:
                    env_map[k] = str(v).strip()
            new_lines = [f"{k}={v}" for k, v in env_map.items()]
            b_path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
            break
        except Exception:
            continue

    # 3. Update local .env files
    for p in _get_candidate_paths():
        if p.name == ".env":
            try:
                env_lines = p.read_text(encoding="utf-8").splitlines() if p.exists() else []
                keys_to_update = set(updates.keys())
                new_lines = [
                    line for line in env_lines
                    if not any(line.strip().startswith(f"{k}=") for k in keys_to_update)
                ]
                for k, v in updates.items():
                    if v is not None:
                        new_lines.append(f"{k}={str(v).strip()}")
                p.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
            except Exception:
                continue

# Automatically load on module import
load_persistent_credentials()

