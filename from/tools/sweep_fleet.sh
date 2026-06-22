#!/usr/bin/env bash
# =============================================================================
# sweep_fleet.sh  —  FLEET trainer / artifact producer for the "from" XAUUSD engine
# -----------------------------------------------------------------------------
# Trains a LARGE grid of walk-forward models across 8x L4 GPUs and lands a
# committee-ready, SAVED artifact per EDGE-positive config in $HOME/registry.
#
# Grid = cartesian product of:
#     ARCH      in {mlp, deep}           (mlp -> 'from walkforward', deep -> 'from wfdeep')
#     HORIZON   in {1024 2048 4096 6144 8192 12288}
#     COST_MULT in {1.0 1.5 2.0}
#     CONF_GATE in {0.50 0.55 0.60}
#     SEED      in {42 137 2718}
#   barrier_k fixed 1.0 ; window 256 for deep, 512 for mlp.
#   2 * 6 * 3 * 3 * 3 = 324 configs.
#
# A WORK QUEUE keeps exactly one container busy per GPU; when a GPU's container
# exits, the next pending config is launched on that GPU. A failed config is
# logged and skipped (it does NOT abort the queue). Ctrl+C stops everything.
#
# Each container writes:
#   /runs/<cfgid>.txt                       full text report (--output)
#   /registry/<artifact-id>/...             model.from|deep.bin + norm1.bin + meta.json
#   /registry/manifest.csv                  one appended row per EDGE-positive run
#
# Usage:
#   bash sweep_fleet.sh /data/FILE.parquet [MAX_SAMPLES] [MAX_STEPS]
#
# Env caps / overrides (for quick tests):
#   MAX_CONFIGS=N        cap how many grid configs to run     (default: all)
#   GRID_HORIZONS="..."  override HORIZON list                (space separated)
#   GRID_COST="..."      override COST_MULT list
#   GRID_GATE="..."      override CONF_GATE list
#   GRID_SEED="..."      override SEED list
#   GRID_ARCH="..."      override ARCH list  (mlp deep)
#   NGPUS=8              number of GPUs to use                 (default: 8)
#   IMAGE=...            docker image                          (default: from-g2-l4:cuda12.8)
#   POLL=30             poll/print interval in seconds         (default: 30)
#   DOCKER="sudo docker" override docker invocation
# =============================================================================
set -uo pipefail

# --- args --------------------------------------------------------------------
DATA="${1:?need data path (container path, e.g. /data/XAUUSD_ticks_all.parquet)}"
MAXS="${2:-40000000}"
STEPS="${3:-8000}"

# --- tunables / environment --------------------------------------------------
IMAGE="${IMAGE:-from-g2-l4:cuda12.8}"
BIN="/workspace/build-l4/from"
NGPUS="${NGPUS:-8}"
POLL="${POLL:-30}"
DOCKER="${DOCKER:-sudo docker}"
MAX_CONFIGS="${MAX_CONFIGS:-0}"          # 0 == unlimited

REGISTRY="${REGISTRY:-$HOME/registry}"
RUNS="$HOME/runs"
mkdir -p "$REGISTRY" "$RUNS"

FAILLOG="$RUNS/fleet_failures.log"
: > "$FAILLOG"

# --- grid lists (overridable via env) ---------------------------------------
ARCHS=(${GRID_ARCH:-mlp deep})
HORIZONS=(${GRID_HORIZONS:-1024 2048 4096 6144 8192 12288})
COSTS=(${GRID_COST:-1.0 1.5 2.0})
GATES=(${GRID_GATE:-0.50 0.55 0.60})
SEEDS=(${GRID_SEED:-42 137 2718})
BARRIER_K="1.0"

# Per-arch fixed knobs.
win_for()   { [ "$1" = deep ] && echo 256 || echo 512; }
batch_for() { [ "$1" = deep ] && echo 4096 || echo 8192; }
cmd_for()   { [ "$1" = deep ] && echo wfdeep || echo walkforward; }

# Canonical config id — MUST match include/io/artifact.hpp make_artifact_id():
#   <arch>_h<H>_cm<CM>_bk<BK>_cg<CG>_s<S>   (cm/bk/cg fixed to 2 decimals)
cfg_id() {
  local arch="$1" H="$2" CM="$3" BK="$4" CG="$5" S="$6"
  printf '%s_h%s_cm%.2f_bk%.2f_cg%.2f_s%s' "$arch" "$H" "$CM" "$BK" "$CG" "$S"
}

# --- build the pending config queue -----------------------------------------
# Each entry: "arch H CM CG S"   (BK is fixed; window/batch derived from arch)
PENDING=()
for arch in "${ARCHS[@]}"; do
  for H in "${HORIZONS[@]}"; do
    for CM in "${COSTS[@]}"; do
      for CG in "${GATES[@]}"; do
        for S in "${SEEDS[@]}"; do
          PENDING+=("$arch $H $CM $CG $S")
        done
      done
    done
  done
done

TOTAL_GRID="${#PENDING[@]}"
if [ "$MAX_CONFIGS" -gt 0 ] && [ "$MAX_CONFIGS" -lt "$TOTAL_GRID" ]; then
  PENDING=("${PENDING[@]:0:$MAX_CONFIGS}")
fi
TOTAL="${#PENDING[@]}"

echo "=============================================================="
echo "[fleet] image=$IMAGE  gpus=$NGPUS  data=$DATA"
echo "[fleet] grid=$TOTAL_GRID configs -> running $TOTAL (MAX_CONFIGS=${MAX_CONFIGS:-0})"
echo "[fleet] arch={${ARCHS[*]}} horizon={${HORIZONS[*]}} cost={${COSTS[*]}}"
echo "[fleet]   gate={${GATES[*]}} seed={${SEEDS[*]}} barrier_k=$BARRIER_K"
echo "[fleet] max_samples=$MAXS max_steps=$STEPS registry=$REGISTRY runs=$RUNS"
echo "=============================================================="

# --- launch one config on a given GPU; echoes the container name -------------
launch_one() {
  local g="$1" entry="$2"
  local arch H CM CG S
  read -r arch H CM CG S <<<"$entry"
  local BK="$BARRIER_K"
  local cmd; cmd="$(cmd_for "$arch")"
  local win; win="$(win_for "$arch")"
  local batch; batch="$(batch_for "$arch")"
  local nm; nm="$(cfg_id "$arch" "$H" "$CM" "$BK" "$CG" "$S")"
  local cpus="$((g*12))-$((g*12+11))"

  # Remove any stale container with this name from a previous run.
  $DOCKER rm -f "$nm" >/dev/null 2>&1 || true

  echo "[launch] GPU$g  $nm  ($cmd win=$win batch=$batch horizon=$H cost=$CM gate=$CG seed=$S)"
  if ! $DOCKER run -d --name "$nm" --gpus "device=${g}" --cpuset-cpus "$cpus" \
        --memory 42g --shm-size 8g --ulimit memlock=-1:-1 \
        --user "$(id -u):$(id -g)" \
        -e OMP_NUM_THREADS=12 \
        -v "$HOME/from:/workspace" -v "$HOME/data:/data" \
        -v "$REGISTRY:/registry" -v "$RUNS:/runs" -w /workspace \
        "$IMAGE" "$BIN" "$cmd" --data "$DATA" \
          --window "$win" --horizon "$H" --cost-mult "$CM" --barrier-k "$BK" \
          --conf-gate "$CG" --seed "$S" --stride 64 --max-samples "$MAXS" \
          --folds 8 --holdout-frac 0.15 --batch-size "$batch" \
          --lr 0.0005 --max-steps "$STEPS" --epochs 14 \
          --model-dir /registry --output "/runs/${nm}.txt" >/dev/null 2>>"$FAILLOG"; then
    echo "[fail] GPU$g $nm: docker run failed to start (see $FAILLOG)" | tee -a "$FAILLOG"
    return 1
  fi
  echo "$nm"
  return 0
}

# --- container status helper -------------------------------------------------
container_status() {  # -> running | exited | gone
  $DOCKER inspect --format '{{.State.Status}}' "$1" 2>/dev/null || echo gone
}
container_exit_code() {
  $DOCKER inspect --format '{{.State.ExitCode}}' "$1" 2>/dev/null || echo "?"
}

# --- per-GPU current container, and bookkeeping ------------------------------
declare -a CUR          # CUR[g] = container name busy on GPU g (empty if idle)
for g in $(seq 0 $((NGPUS-1))); do CUR[$g]=""; done
NEXT=0                  # index into PENDING of the next config to launch
COMPLETED=0
FAILED=0
EDGE_YES=0

# --- reap a finished container: log result, free the GPU ---------------------
reap() {
  local g="$1" nm="${CUR[$1]}"
  [ -z "$nm" ] && return 0
  local code; code="$(container_exit_code "$nm")"
  local edge; edge="$(grep -h '^EDGE:' "$RUNS/${nm}.txt" 2>/dev/null | tail -1)"
  if [ "$code" = "0" ]; then
    COMPLETED=$((COMPLETED+1))
    if printf '%s' "$edge" | grep -qi 'yes'; then EDGE_YES=$((EDGE_YES+1)); fi
    echo "[done] GPU$g $nm exit=0  ${edge:-<no EDGE line>}"
  else
    COMPLETED=$((COMPLETED+1))
    FAILED=$((FAILED+1))
    echo "[fail] GPU$g $nm exit=$code (skipping; see $RUNS/${nm}.txt)" | tee -a "$FAILLOG"
    $DOCKER logs --tail 20 "$nm" >>"$FAILLOG" 2>&1 || true
  fi
  $DOCKER rm "$nm" >/dev/null 2>&1 || true
  CUR[$g]=""
}

# --- graceful shutdown on Ctrl+C ---------------------------------------------
STOP=0
shutdown() {
  STOP=1
  echo; echo "[fleet] Ctrl+C — stopping all running containers ..."
  for g in $(seq 0 $((NGPUS-1))); do
    nm="${CUR[$g]}"
    [ -n "$nm" ] && { $DOCKER stop -t 5 "$nm" >/dev/null 2>&1 || true; $DOCKER rm -f "$nm" >/dev/null 2>&1 || true; }
  done
}
trap shutdown INT TERM

# --- seed the queue: fill every GPU ------------------------------------------
for g in $(seq 0 $((NGPUS-1))); do
  [ "$NEXT" -ge "$TOTAL" ] && break
  nm="$(launch_one "$g" "${PENDING[$NEXT]}" | tail -1)"
  if [ -n "$nm" ] && [ "$nm" != "1" ]; then CUR[$g]="$nm"; else FAILED=$((FAILED+1)); COMPLETED=$((COMPLETED+1)); fi
  NEXT=$((NEXT+1))
done

# --- main work-queue loop ----------------------------------------------------
last_print=0
while :; do
  [ "$STOP" -eq 1 ] && break

  # Reap finished GPUs and launch the next pending config there.
  running=0
  for g in $(seq 0 $((NGPUS-1))); do
    nm="${CUR[$g]}"
    if [ -n "$nm" ]; then
      st="$(container_status "$nm")"
      if [ "$st" = "running" ]; then
        running=$((running+1))
      else
        reap "$g"
        if [ "$NEXT" -lt "$TOTAL" ] && [ "$STOP" -eq 0 ]; then
          nm2="$(launch_one "$g" "${PENDING[$NEXT]}" | tail -1)"
          if [ -n "$nm2" ] && [ "$nm2" != "1" ]; then CUR[$g]="$nm2"; running=$((running+1)); else FAILED=$((FAILED+1)); COMPLETED=$((COMPLETED+1)); fi
          NEXT=$((NEXT+1))
        fi
      fi
    fi
  done

  # Done when nothing running and nothing pending.
  if [ "$running" -eq 0 ] && [ "$NEXT" -ge "$TOTAL" ]; then break; fi

  # Progress print every POLL seconds.
  now="$(date +%s)"
  if [ "$((now - last_print))" -ge "$POLL" ]; then
    last_print="$now"
    busy=()
    for g in $(seq 0 $((NGPUS-1))); do [ -n "${CUR[$g]}" ] && busy+=("g$g:${CUR[$g]}"); done
    echo "--- $(date -u +%H:%M:%S)  running=$running/$NGPUS  completed=$COMPLETED/$TOTAL  edge_yes=$EDGE_YES  failed=$FAILED  queued=$((TOTAL-NEXT)) ---"
    [ "${#busy[@]}" -gt 0 ] && printf '    [busy] %s\n' "${busy[*]}"
    nvidia-smi --query-gpu=index,utilization.gpu,memory.used,power.draw --format=csv,noheader 2>/dev/null \
      | sed 's/^/    gpu /' || echo "    (nvidia-smi unavailable)"
  fi

  sleep 2
done

# --- final reap (in case loop broke right after exits) -----------------------
if [ "$STOP" -eq 0 ]; then
  for g in $(seq 0 $((NGPUS-1))); do
    nm="${CUR[$g]}"
    [ -n "$nm" ] && [ "$(container_status "$nm")" != "running" ] && reap "$g"
  done
fi

# --- summary -----------------------------------------------------------------
echo
echo "=============================================================="
echo "[fleet] finished: completed=$COMPLETED/$TOTAL  edge_yes=$EDGE_YES  failed=$FAILED"
artifacts="$(find "$REGISTRY" -maxdepth 1 -mindepth 1 -type d 2>/dev/null | wc -l | tr -d ' ')"
echo "[fleet] $artifacts artifact dir(s) landed in $REGISTRY"
[ -s "$FAILLOG" ] && echo "[fleet] failures logged to $FAILLOG"
echo "=============================================================="

# --- leaderboard via registry_report.py --------------------------------------
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "$REGISTRY/manifest.csv" ]; then
  python3 "$HERE/registry_report.py" "$REGISTRY/manifest.csv" || \
    echo "[fleet] registry_report.py failed; manifest at $REGISTRY/manifest.csv"
else
  echo "[fleet] no manifest.csv yet (no EDGE-positive artifacts saved)."
fi
