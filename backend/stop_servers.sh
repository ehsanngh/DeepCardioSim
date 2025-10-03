#!/usr/bin/env bash
set -euo pipefail

# Load env
set -a
. "$(dirname "$0")/.env.path"
set +a

echo "[cleanup] Stopping all services..."

# Stop uvicorn
if [ -f "$LOG_DIR/uvicorn.pid" ]; then
    kill $(cat "$LOG_DIR/uvicorn.pid") 2>/dev/null || true
    rm "$LOG_DIR/uvicorn.pid"
fi

# Stop RQ workers (EF and CRT)
for pidfile in "$LOG_DIR"/*.pid; do
    [ -f "$pidfile" ] || continue
    kill $(cat "$pidfile") 2>/dev/null || true
    rm "$pidfile"
done


# Stop Redis instance
echo "[cleanup] Stopping Redis..."
singularity instance stop redis 2>/dev/null || true

echo "All services stopped and cleaned up"