#!/usr/bin/env bash
# Keeps one LIBERO client running until all episodes are recorded. A client whose policy input
# has not changed for STALL_SECONDS gets SIGUSR1 (stack dump to its log), is killed and restarted;
# completed episodes are skipped on restart. Each attempt waits until the VLA server answers,
# so a server paused by gpu_keeper.sh does not burn restarts. Usage: supervise_libero.sh OUT TRIALS METHOD
set -uo pipefail
source "$(dirname "$0")/env.sh"
export PYTHONPATH=$RM/LIBERO
out=$1 trials=$2 method=$3
STALL_SECONDS=${STALL_SECONDS:-900} MAX_RESTARTS=${MAX_RESTARTS:-20}
total=$((trials * 10)) dir=$out/$method log=$out/$method.log
mkdir -p "$dir"
for attempt in $(seq 0 "$MAX_RESTARTS"); do
  done_n=$(wc -l < "$dir/episodes.jsonl" 2>/dev/null || echo 0)
  [ "$done_n" -ge "$total" ] && { echo "$(date -u +%FT%TZ) $method complete ($done_n)" >> "$log"; exit 0; }
  until curl -s -m 5 localhost:3201/health | grep -q true; do sleep 30; done
  echo "$(date -u +%FT%TZ) supervisor: start attempt $attempt at $done_n/$total" >> "$log"
  touch "$dir/policy_input.png"
  "$RM/envs/libero/bin/python" "$RM/selection/experiments/online/libero/run_eval.py" \
    --method "$method" --n-pool 8 --trials "$trials" --out "$dir" \
    --norm-stats "$RM/stats/libero_10.json" --vla-port 3201 >> "$log" 2>&1 < /dev/null &
  pid=$!
  while kill -0 "$pid" 2>/dev/null; do
    sleep 60
    last=$(stat -c %Y "$dir/policy_input.png" 2>/dev/null || date +%s)
    if [ $(( $(date +%s) - last )) -gt "$STALL_SECONDS" ]; then
      echo "$(date -u +%FT%TZ) supervisor: stalled > ${STALL_SECONDS}s, dumping stack and restarting" >> "$log"
      kill -USR1 "$pid"; sleep 5; kill "$pid"; sleep 5; kill -9 "$pid" 2>/dev/null
      touch "$dir/policy_input.png"
    fi
  done
done
echo "$(date -u +%FT%TZ) supervisor: giving up after $MAX_RESTARTS restarts" >> "$log"
