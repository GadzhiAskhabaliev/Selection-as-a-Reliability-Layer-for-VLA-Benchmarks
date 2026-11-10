#!/usr/bin/env bash
# Yields the shared GPU: if free memory drops below MIN_FREE_MIB, stop our VLA servers.
MIN_FREE_MIB=${MIN_FREE_MIB:-3000}
LOG=${RM:-$HOME/rm_bench}/logs/gpu_guard.log
echo "$(date -u +%FT%TZ) guard started, threshold ${MIN_FREE_MIB} MiB" >> "$LOG"
while pgrep -f "[v]la_server.py" > /dev/null; do
  free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
  if [ -n "$free" ] && [ "$free" -lt "$MIN_FREE_MIB" ]; then
    echo "$(date -u +%FT%TZ) free=${free} MiB < ${MIN_FREE_MIB}: stopping our servers" >> "$LOG"
    nvidia-smi --query-compute-apps=pid,used_memory,process_name --format=csv,noheader >> "$LOG" 2>&1
    pkill -f "[v]la_server.py"
    break
  fi
  sleep 5
done
echo "$(date -u +%FT%TZ) guard exiting" >> "$LOG"
