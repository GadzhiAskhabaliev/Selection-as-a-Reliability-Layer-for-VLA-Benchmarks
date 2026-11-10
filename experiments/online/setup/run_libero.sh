#!/usr/bin/env bash
# Paired LIBERO-10 runs: identical tasks and initial states for every method, one shared VLA server
# (start it first with `serve.sh vla_libero`). Usage: run_libero.sh OUT TRIALS METHOD [METHOD ...]
set -euo pipefail
source "$(dirname "$0")/env.sh"
export PYTHONPATH=$RM/LIBERO
out=$1 trials=$2; shift 2
mkdir -p "$out"
for method in "$@"; do
  setsid nohup "$RM/envs/libero/bin/python" "$RM/selection/experiments/online/libero/run_eval.py" \
    --method "$method" --n-pool 8 --trials "$trials" --out "$out/$method" \
    --norm-stats "$RM/stats/libero_10.json" --vla-port 3201 > "$out/$method.log" 2>&1 < /dev/null &
  echo "started $method (pid $!)"
done
