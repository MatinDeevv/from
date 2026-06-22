#!/usr/bin/env bash
# 8-GPU sweep of the RAW-WINDOW deep model (wfdeep): one L4 per config.
# Usage: bash sweep_deep.sh /data/FILE.parquet [MAX_SAMPLES] [MAX_STEPS]
set -uo pipefail
DATA="${1:?need data path}"
MAXS="${2:-60000000}"
STEPS="${3:-6000}"
BATCH="${BATCH:-4096}"
WIN="${WIN:-256}"
IMAGE="${IMAGE:-from-g2-l4:cuda12.8}"
BIN="/workspace/build-l4/from"
RUNS="$HOME/runs"; mkdir -p "$RUNS"

# "horizon cost_mult barrier_k conf_gate"
CONFIGS=(
  "2048 1.5 1.0 0.55"
  "4096 1.5 1.0 0.55"
  "4096 2.0 1.0 0.60"
  "8192 1.5 1.0 0.55"
  "8192 2.0 1.0 0.60"
  "12288 1.5 1.0 0.55"
  "16384 2.0 1.0 0.60"
  "6144 1.5 1.0 0.55"
)
names=()
for g in $(seq 0 7); do
  cfg=(${CONFIGS[$g]}); H=${cfg[0]}; CM=${cfg[1]}; BK=${cfg[2]}; CG=${cfg[3]}
  nm="deep_g${g}_h${H}_cm${CM}"; names+=("$nm")
  cpus="$((g*12))-$((g*12+11))"
  echo "[launch] GPU$g $nm horizon=$H cost_mult=$CM gate=$CG"
  sudo docker run -d --name "$nm" --gpus "device=${g}" --cpuset-cpus "$cpus" \
    --memory 42g --shm-size 8g --ulimit memlock=-1:-1 --user "$(id -u):$(id -g)" \
    -e OMP_NUM_THREADS=12 \
    -v "$HOME/from:/workspace" -v "$HOME/data:/data" -v "$RUNS:/runs" -w /workspace \
    "$IMAGE" "$BIN" wfdeep --data "$DATA" \
      --window "$WIN" --horizon "$H" --cost-mult "$CM" --barrier-k "$BK" --conf-gate "$CG" \
      --stride 64 --max-samples "$MAXS" --folds 8 --holdout-frac 0.15 \
      --batch-size "$BATCH" --lr 0.0005 --max-steps "$STEPS" --epochs 14 \
      --output "/runs/${nm}.txt" >/dev/null
done
echo "[deep sweep] launched; polling 30s ..."
while :; do
  r=0; for nm in "${names[@]}"; do [ "$(sudo docker inspect -f '{{.State.Status}}' "$nm" 2>/dev/null)" = running ] && r=$((r+1)); done
  echo "--- $(date -u +%H:%M:%S) running=$r/8 ---"
  nvidia-smi --query-gpu=index,utilization.gpu,memory.used,power.draw --format=csv,noheader | sed 's/^/  gpu /'
  [ "$r" -eq 0 ] && break; sleep 30
done
echo; echo "============ DEEP LEADERBOARD (after-cost holdout) ============"
for nm in "${names[@]}"; do
  printf '%-26s %s\n' "$nm" "$(grep -h '^EDGE:' "$RUNS/${nm}.txt" 2>/dev/null | tail -1 || echo '<none>')"
done
echo "==============================================================="
for nm in "${names[@]}"; do sudo docker rm "$nm" >/dev/null 2>&1 || true; done
