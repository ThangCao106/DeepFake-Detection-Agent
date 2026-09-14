#!/usr/bin/env bash
set -euo pipefail

echo "============================================"
echo "  Observability Agent (local test only)"
echo "  Chi nghe tren 127.0.0.1 - khong mo LAN"
echo "  Ctrl+C de tat"
echo "============================================"

cd "$(dirname "$0")"
exec uv run uvicorn app.main:app --host 127.0.0.1 --port 8787
