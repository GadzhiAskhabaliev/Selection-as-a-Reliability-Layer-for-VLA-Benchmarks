#!/usr/bin/env bash
# Runs one GPU job only when the shared GPU has room and none of our other GPU work is active;
# kills it if free memory falls below MIN_FREE_MIB and retries later from scratch.
# Usage: gpu_job.sh LOG -- COMMAND [ARGS...]
set -uo pipefail
source "$(dirname "$0")/env.sh"
log=$1; shift; [ "$1" = "--" ] && shift
START_FREE_MIB=${START_FREE_MIB:-35000} MIN_FREE_MIB=${MIN_FREE_MIB:-3000} MAX_ATTEMPTS=${MAX_ATTEMPTS:-5}
note() { echo "$(date -u +%FT%TZ) gpu_job: $*" >> "$log"; }
free_mib() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' '; }
busy() { pgrep -f "online/setup/[s]upervise_libero" > /dev/null || pgrep -f "[v]la_server.py" > /dev/null; }

for attempt in $(seq 1 "$MAX_ATTEMPTS"); do
  until ! busy && [ "$(free_mib)" -ge "$START_FREE_MIB" ]; do sleep 60; done
  note "attempt $attempt: starting with $(free_mib) MiB free"
  "$@" >> "$log" 2>&1 < /dev/null &
  pid=$! guarded=0
  while kill -0 "$pid" 2>/dev/null; do
    sleep 5
    free=$(free_mib)
    if [ -n "$free" ] && [ "$free" -lt "$MIN_FREE_MIB" ]; then
      note "free=${free} MiB: stopping job to yield the GPU"
      nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader >> "$log" 2>&1
      kill "$pid"; sleep 10; kill -9 "$pid" 2>/dev/null; guarded=1
    fi
  done
  wait "$pid"; rc=$?
  [ "$rc" -eq 0 ] && { note "completed"; exit 0; }
  [ "$guarded" -eq 0 ] && { note "failed with exit code $rc (not a GPU-memory stop); not retrying"; exit "$rc"; }
done
note "gave up after $MAX_ATTEMPTS attempts"
exit 1
