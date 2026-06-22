#!/usr/bin/env bash
# 8-GPU walk-forward sweep: one L4 per (horizon, cost_mult, barrier_k, conf_gate) config.
# Each runs the honest purged walk-forward + holdout; we collect the EDGE verdicts.
#
# Usage: bash sweep_wf.sh /data/XAUUSD_ticks_all.parquet [MAX_SAMPLES] [MAX_STEPS]
set -uo pipefail

DATA="${1:?need data path (container path, e.g. /data/XAUUSD_ticks_all.parquet)}"
MAXS="${2:-40000000}"
STEPS="${3:-8000}"
BATCH="${BATCH:-8192}"
IMAGE="${IMAGE:-from-g2-l4:cuda12.8}"
BIN="/workspace/build-l4/from"
RUNS="$HOME/runs"
mkdir -p "$RUNS"

# config: "horizon cost_mult barrier_k conf_gate"  (one per GPU 0..7)
# Break-even region focus: long horizons where moves clear cost; vary gate + barrier.
CONFIGS=(
  "8192 1.0 1.0 0.50"
  "8192 1.5 1.0 0.55"
  "10240 1.0 1.0 0.50"
  "10240 1.5 1.0 0.55"
  "12288 1.0 1.0 0.50"
  "12288 1.5 1.0 0.55"
  "8192 2.0 1.0 0.60"
  "12288 2.0 1.0 0.60"
)

names=()
for g in $(seq 0 7); do
  cfg=(${CONFIGS[$g]}); H=${cfg[0]}; CM=${cfg[1]}; BK=${cfg[2]}; CG=${cfg[3]}
  nm="wf_g${g}_h${H}_cm${CM}"
  names+=("$nm")
  cpus="$((g*12))-$((g*12+11))"
  echo "[launch] GPU$g $nm  horizon=$H cost_mult=$CM barrier_k=$BK conf_gate=$CG"
  sudo docker run -d --name "$nm" --gpus "device=${g}" --cpuset-cpus "$cpus" \
    --memory 42g --shm-size 8g --ulimit memlock=-1:-1 \
    --user "$(id -u):$(id -g)" \
    -e OMP_NUM_THREADS=12 \
    -v "$HOME/from:/workspace" -v "$HOME/data:/data" -v "$RUNS:/runs" -w /workspace \
    "$IMAGE" "$BIN" walkforward --data "$DATA" \
      --horizon "$H" --cost-mult "$CM" --barrier-k "$BK" --conf-gate "$CG" \
      --stride 64 --max-samples "$MAXS" --folds 8 --holdout-frac 0.15 \
      --batch-size "$BATCH" --lr 0.0005 --max-steps "$STEPS" --epochs 14 \
      --output "/runs/${nm}.txt" >/dev/null
done

echo "[sweep] 8 workers launched; polling every 30s ..."
while :; do
  running=0
  for nm in "${names[@]}"; do
    st="$(sudo docker inspect --format '{{.State.Status}}' "$nm" 2>/dev/null || echo gone)"
    [ "$st" = "running" ] && running=$((running+1))
  done
  echo "--- $(date -u +%H:%M:%S)  running=$running/8 ---"
  nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader | sed 's/^/  gpu /'
  [ "$running" -eq 0 ] && break
  sleep 30
done

echo; echo "================ LEADERBOARD (after-cost holdout) ================"
for nm in "${names[@]}"; do
  line="$(grep -h '^EDGE:' "$RUNS/${nm}.txt" 2>/dev/null | tail -1)"
  printf '%-24s %s\n' "$nm" "${line:-<no EDGE line; check $RUNS/${nm}.txt>}"
done
echo "=================================================================="
for nm in "${names[@]}"; do sudo docker rm "$nm" >/dev/null 2>&1 || true; done
