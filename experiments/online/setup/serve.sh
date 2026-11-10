#!/usr/bin/env bash
# Starts one model service in the foreground. Usage: serve.sh {vla_bridge|vla_libero|verifier}
set -euo pipefail
source "$(dirname "$0")/env.sh"
ONLINE=$RM/selection/experiments/online
CHECK=$RM/stats/check.png
[ -f "$CHECK" ] || "$RM/envs/vla/bin/python" -c "import numpy as np; from PIL import Image; \
Image.fromarray((np.random.default_rng(0).random((224, 224, 3)) * 255).astype('uint8')).save('$CHECK')"
case "$1" in
  vla_bridge)
    exec "$RM/envs/vla/bin/python" "$ONLINE/vla_server.py" --model openvla/openvla-7b \
      --unnorm-key bridge_orig --port 3200 --seed 1 --check-image "$CHECK" ;;
  vla_libero)
    exec "$RM/envs/vla/bin/python" "$ONLINE/vla_server.py" --model openvla/openvla-7b-finetuned-libero-10 \
      --unnorm-key libero_10 --port 3201 --seed 7 --check-image "$CHECK" ;;
  verifier)
    cd "$RM/RoboMonkey/monkey-verifier/src"
    MODEL_DIR=$RM/models/monkey-verifier-7b exec "$RM/envs/verifier/bin/python" infer_server.py ;;
  *) echo "unknown service $1"; exit 2 ;;
esac
