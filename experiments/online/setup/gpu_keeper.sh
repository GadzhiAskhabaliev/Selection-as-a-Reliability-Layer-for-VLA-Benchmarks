#!/usr/bin/env bash
# Runs the VLA server only while the shared GPU has room for it, for as long as LIBERO
# supervisors are active. Stops the server when free memory falls below MIN_FREE_MIB and
# starts it again once START_FREE_MIB is available. Usage: gpu_keeper.sh SERVICE PORT
set -uo pipefail
source "$(dirname "$0")/env.sh"
service=$1 port=$2
MIN_FREE_MIB=${MIN_FREE_MIB:-3000} START_FREE_MIB=${START_FREE_MIB:-25000}
LOG=$RM/logs/gpu_keeper.log
log() { echo "$(date -u +%FT%TZ) $*" >> "$LOG"; }
free_mib() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' '; }
server_pid() { pgrep -f "[v]la_server.py.*--port $port" | head -1; }

log "keeper started: $service on $port, stop < ${MIN_FREE_MIB} MiB, start >= ${START_FREE_MIB} MiB"
idle=0
while true; do
  free=$(free_mib)
  pid=$(server_pid)
  if [ -n "$pid" ] && [ -n "$free" ] && [ "$free" -lt "$MIN_FREE_MIB" ]; then
    log "free=${free} MiB: stopping server $pid"
    nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader >> "$LOG" 2>&1
    kill "$pid"; sleep 10; kill -9 "$pid" 2>/dev/null
  elif [ -z "$pid" ] && [ -n "$free" ] && [ "$free" -ge "$START_FREE_MIB" ]; then
    log "free=${free} MiB: starting $service"
    setsid nohup "$(dirname "$0")/serve.sh" "$service" >> "$RM/logs/$service.log" 2>&1 < /dev/null &
    sleep 60
  fi
  if pgrep -f "[s]upervise_libero.sh" > /dev/null; then idle=0; else idle=$((idle + 1)); fi
  if [ "$idle" -ge 4 ]; then
    pid=$(server_pid)
    [ -n "$pid" ] && kill "$pid"
    log "no active supervisors: server stopped, keeper exiting"
    exit 0
  fi
  sleep 15
done
