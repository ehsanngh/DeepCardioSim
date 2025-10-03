#!/usr/bin/env bash
set -euo pipefail

# Load env
set -a
. "$(dirname "$0")/.env.path"
set +a

mkdir -p "$LOG_DIR"

TOTAL_CORES=$((NUM_WORKERS * NUM_PROCS_PER_WORKER))
GPUS="${NUM_GPUS:-0}"
EF_TOTAL=$((TOTAL_CORES - GPUS))
(( EF_TOTAL < 1 )) && EF_TOTAL=1

echo "[redis] starting singularity instance..."
singularity instance start \
  --bind "$REDIS_DATA_PATH:/data" \
  --bind "$REDIS_CONF_PATH:/etc/redis/redis.conf" \
  "$REDIS_CONTAINER_PATH" redis 2>>"$LOG_DIR/redis.err.log" || true

singularity exec instance://redis redis-server \
  --daemonize yes >>"$LOG_DIR/redis.out.log" 2>>"$LOG_DIR/redis.err.log"

for ((i=0; i<EF_TOTAL; i++)); do
  name="ef-$i"
  echo "[rq] starting EF worker $name"
  nohup uv run rq worker ef --url "$REDIS_URL" --name "$name" \
    >>"$LOG_DIR/${name}.log" 2>&1 &
  echo $! >"$LOG_DIR/${name}.pid"
done

for ((gid=0; gid<GPUS; gid++)); do
  name="crt-gpu${gid}"
  echo "[rq] starting CRT worker on GPU $gid"
  CUDA_VISIBLE_DEVICES="${gid}" \
  nohup uv run rq worker crt --url "${REDIS_URL}" --name "$name" \
      >>"$LOG_DIR/${name}.log" 2>&1 &
  echo $! >"$LOG_DIR/${name}.pid"
done

echo "[uvicorn] starting api with ${NUM_WORKERS} workers"

nohup uvicorn main:app --host 127.0.0.1 --port 8000 --workers "${NUM_WORKERS}" \
  --no-access-log --log-level warning \
  >>"$LOG_DIR/uvicorn.log" 2>&1 &
echo $! >"$LOG_DIR/uvicorn.pid"

find "$LOG_DIR" -type f -name '*.log*' -mtime +10 -delete || true
echo "All services launched. Logs in $LOG_DIR"