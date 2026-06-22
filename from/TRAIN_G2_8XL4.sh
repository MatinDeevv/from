#!/usr/bin/env bash
set -Eeuo pipefail

# G2-standard-96 deployment launcher:
#   96 vCPU, 384 GiB RAM, 8x NVIDIA L4, Ubuntu Minimal 26.04 host.
# CUDA builds/runs in an Ubuntu 24.04 CUDA container so the new host OS does
# not need a matching host CUDA toolkit. Only the NVIDIA kernel driver is host-side.
#
# Training proceeds in three stages, designed to fit a ~6h GPU window:
#   STAGE 0  Canary gate    ~2 min   GPU 0 only. Short smoke run; aborts the whole
#                                    launcher unless the binary trains cleanly.
#   STAGE 1  Config sweep   ~75 min  8 GPUs in parallel, one (threshold,horizon,
#                                    stride,lr,max_samples) config each. The best
#                                    after-cost edge with positive Kelly wins.
#   STAGE 2  Ensemble       rest     8 GPUs in parallel, the winning config trained
#                                    under 8 different seeds for a seed ensemble.
# Each distinct (window,stride,horizon,threshold,max_samples) config derives its
# own cache file, so configs never collide and each builds its cache on first use.

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DATA_PATH="${1:-${DATA_PATH:-${ROOT_DIR}/XAUUSD_ticks_all.parquet}}"
MAX_SAMPLES="${MAX_SAMPLES:-18000000}"
BATCH_SIZE="${BATCH_SIZE:-65536}"
MAX_STEPS="${MAX_STEPS:-1000000}"
LEARNING_RATE="${LEARNING_RATE:-0.00005}"
VALIDATE_EVERY="${VALIDATE_EVERY:-10000}"
SAVE_EVERY="${SAVE_EVERY:-100000}"
IMAGE_NAME="${IMAGE_NAME:-from-g2-l4:cuda12.8}"
BUILD_DIR="${ROOT_DIR}/build-g2-l4"
RUN_ROOT="${RUN_ROOT:-${ROOT_DIR}/runs/g2-8xl4-$(date -u +%Y%m%dT%H%M%SZ)}"
GPU_COUNT=8
CPU_THREADS_PER_GPU=12
WORKER_MEMORY="42g"
SEEDS=(42 137 2718 31415 104729 524287 999983 15485863)

# ----------------------------------------------------------------------------
# Stage 0 (canary gate) budget: a short smoke run on GPU 0 only.
CANARY_MAX_STEPS=3000
CANARY_VALIDATE_EVERY=1000
CANARY_MAX_SAMPLES=4000000
CANARY_EXPLODE_LIMIT=20

# Stage 1 (config sweep) budget: same for every worker.
SWEEP_MAX_STEPS=60000
SWEEP_VALIDATE_EVERY=5000
SWEEP_SAVE_EVERY=30000
SWEEP_BATCH_SIZE=65536

# Stage 1 per-GPU grid. One entry per GPU 0..7. Each row is a single string of
# space-separated fields in this exact order:
#   direction_threshold horizon stride lr max_samples
# Values are copied verbatim from the deployment spec.
SWEEP_GRID=(
    "0.5  256 64  0.0005 12000000"   # gpu0  baseline-fix: 2.0->0.5 at legacy horizon
    "0.3  128 32  0.0006 12000000"   # gpu1  short-horizon momentum, dense stride
    "1    512 128 0.0004 10000000"   # gpu2  trend capture, long horizon, wide stride
    "0.5  128 64  0.0005 12000000"   # gpu3  mid-band short horizon, A/B vs gpu0
    "0.75 256 64  0.0004 12000000"   # gpu4  cost-aware band ~ round-trip floor
    "0.3  256 48  0.0006 14000000"   # gpu5  max-signal, very low band, high samples
    "1.5  384 128 0.0003 10000000"   # gpu6  selective large-move, sparse-but-clean
    "0.5  256 64  0.0008 12000000"   # gpu7  lr-stress twin of gpu0 (~1.6x lr)
)

# Stage 2 (ensemble winner) budget. The winning (threshold,horizon,stride,lr)
# is discovered at runtime; only the budget is fixed here.
ENSEMBLE_MAX_SAMPLES=18000000
ENSEMBLE_MAX_STEPS=300000
ENSEMBLE_VALIDATE_EVERY=10000
ENSEMBLE_SAVE_EVERY=100000
ENSEMBLE_BATCH_SIZE=65536
# Eight distinct seeds for the final ensemble (same labels, different inits).
ENSEMBLE_SEEDS=(42 137 2718 31415 104729 524287 999983 15485863)

log() { printf '[%s] %s\n' "$(date -u +%H:%M:%S)" "$*"; }
die() { printf '[FATAL] %s\n' "$*" >&2; exit 1; }
docker_cmd() { sudo docker "$@"; }

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    cat <<'EOF'
Usage: bash TRAIN_G2_8XL4.sh /absolute/path/XAUUSD_ticks_all.parquet

Optional environment overrides:
  MAX_SAMPLES=18000000 BATCH_SIZE=65536 MAX_STEPS=1000000
  LEARNING_RATE=0.00005 VALIDATE_EVERY=10000 SAVE_EVERY=100000
  RUN_ROOT=/path/to/output

First run: installs the host NVIDIA driver and exits for a reboot if needed.
After reboot, run the same command again. It then installs Docker/NVIDIA runtime,
builds for L4 compute capability 8.9, and runs three stages:
  Stage 0  a ~2 min canary smoke run on GPU 0 that gates the rest of the launch;
  Stage 1  an 8-GPU config sweep over (threshold,horizon,stride,lr) configs;
  Stage 2  an 8-seed ensemble of the winning config.
Outputs land under RUN_ROOT (sweep/, ensemble/, leaderboard.txt).
EOF
    exit 0
fi

[[ -r /etc/os-release ]] || die 'This launcher requires Linux.'
# shellcheck disable=SC1091
source /etc/os-release
[[ "${ID:-}" == "ubuntu" ]] || die "Expected Ubuntu, found ${ID:-unknown}."
if [[ "${VERSION_ID:-}" != "26.04" ]]; then
    log "Warning: tuned for Ubuntu Minimal 26.04; detected ${VERSION_ID:-unknown}."
fi

install_base_packages() {
    log 'Installing host prerequisites.'
    sudo apt-get update
    sudo apt-get install -y --no-install-recommends \
        ca-certificates curl gpg linux-headers-"$(uname -r)" \
        ubuntu-drivers-common docker.io util-linux procps sysstat
}

install_base_packages

if ! command -v nvidia-smi >/dev/null 2>&1 || ! nvidia-smi >/dev/null 2>&1; then
    log 'Installing Ubuntu 26.04 NVIDIA 595 server driver and utilities.'
    sudo ubuntu-drivers install --gpgpu
    sudo apt-get install -y nvidia-utils-595-server
    cat <<'EOF'

[REBOOT REQUIRED] The NVIDIA kernel driver was installed.
Run: sudo reboot
Then rerun this exact training command. No training has started yet.
EOF
    exit 20
fi

install_nvidia_container_runtime() {
    if command -v nvidia-ctk >/dev/null 2>&1; then
        return
    fi
    log 'Installing NVIDIA Container Toolkit.'
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
        | sudo gpg --dearmor --yes -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
    curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
        | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
        | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list >/dev/null
    sudo apt-get update
    sudo apt-get install -y nvidia-container-toolkit
    sudo nvidia-ctk runtime configure --runtime=docker
    sudo systemctl restart docker
}

install_nvidia_container_runtime

mapfile -t GPU_NAMES < <(nvidia-smi --query-gpu=name --format=csv,noheader)
[[ "${#GPU_NAMES[@]}" -eq "$GPU_COUNT" ]] \
    || die "Expected 8 GPUs; nvidia-smi reports ${#GPU_NAMES[@]}."
for i in "${!GPU_NAMES[@]}"; do
    [[ "${GPU_NAMES[$i]}" == *L4* ]] || die "GPU ${i} is not an L4: ${GPU_NAMES[$i]}"
done

CPU_COUNT="$(nproc --all)"
MEM_KIB="$(awk '/MemTotal:/ {print $2}' /proc/meminfo)"
[[ "$CPU_COUNT" -ge 96 ]] || die "Expected at least 96 vCPUs; found ${CPU_COUNT}."
[[ "$MEM_KIB" -ge 360000000 ]] || die "Expected roughly 384 GiB RAM; found $((MEM_KIB / 1024 / 1024)) GiB."
[[ -f "$DATA_PATH" ]] || die "Dataset not found: ${DATA_PATH}"

DATA_PATH="$(readlink -f -- "$DATA_PATH")"
DATA_DIR="$(dirname -- "$DATA_PATH")"
DATA_FILE="$(basename -- "$DATA_PATH")"
mkdir -p "$BUILD_DIR" "$RUN_ROOT"
RUN_ROOT="$(readlink -f -- "$RUN_ROOT")"

log 'Enabling persistence mode and performance-oriented host settings.'
sudo nvidia-smi -pm 1 >/dev/null || true
sudo sysctl -q -w vm.swappiness=1 || true
for governor in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
    [[ -e "$governor" ]] && echo performance | sudo tee "$governor" >/dev/null || true
done

log "Building CUDA environment image ${IMAGE_NAME}."
docker_cmd build --pull -f "${ROOT_DIR}/Dockerfile.g2-l4" -t "$IMAGE_NAME" "$ROOT_DIR"

log 'Compiling Release binary for NVIDIA L4 (SM 8.9) with 96-way build parallelism.'
docker_cmd run --rm \
    --user "$(id -u):$(id -g)" \
    --volume "${ROOT_DIR}:/workspace" \
    --workdir /workspace \
    "$IMAGE_NAME" \
    bash -lc "cmake -S /workspace -B /workspace/$(basename "$BUILD_DIR") -G Ninja \
      -DCMAKE_BUILD_TYPE=Release -DFROM_CUDA=ON -DFROM_NATIVE_ARCH=ON \
      -DCMAKE_CUDA_ARCHITECTURES=89 && \
      cmake --build /workspace/$(basename "$BUILD_DIR") --parallel 96"

TRAIN_BIN="${BUILD_DIR}/from"
[[ -x "$TRAIN_BIN" ]] || die "Build did not produce ${TRAIN_BIN}."
CONTAINER_TRAIN_BIN="/workspace/$(basename "$BUILD_DIR")/from"

# ============================================================================
# Worker bookkeeping + signal-safe cleanup. WORKER_NAMES/WORKER_DIRS hold the
# currently-active stage's detached containers and are reset between stages.
# ============================================================================
WORKER_NAMES=()
WORKER_DIRS=()
cleanup() {
    trap - INT TERM EXIT
    if ((${#WORKER_NAMES[@]})); then
        log 'Stopping training workers.'
        docker_cmd stop --time 30 "${WORKER_NAMES[@]}" >/dev/null 2>&1 || true
    fi
}
on_signal() {
    cleanup
    exit 130
}
trap on_signal INT TERM
trap cleanup EXIT

# launch_worker GPU CPU_FIRST CPU_LAST NAME DIR -- <train args...>
# Reuses the existing detached-docker pattern: one L4 pinned to a vCPU range and
# 42g RAM, 8g shm, unlimited memlock, OMP env, workspace(ro)+data+per-worker
# /runs volumes, local log driver. Everything after "--" is the train command.
launch_worker() {
    local gpu="$1" cpu_first="$2" cpu_last="$3" name="$4" dir="$5"
    shift 5
    [[ "$1" == "--" ]] && shift
    mkdir -p "$dir"
    docker_cmd run --detach \
        --name "$name" \
        --gpus "device=${gpu}" \
        --cpuset-cpus "${cpu_first}-${cpu_last}" \
        --memory "$WORKER_MEMORY" \
        --memory-swap "$WORKER_MEMORY" \
        --shm-size 8g \
        --ulimit memlock=-1:-1 \
        --env OMP_NUM_THREADS="$CPU_THREADS_PER_GPU" \
        --env OMP_PROC_BIND=close \
        --env OMP_PLACES=cores \
        --env MALLOC_ARENA_MAX=4 \
        --volume "${ROOT_DIR}:/workspace:ro" \
        --volume "${DATA_DIR}:/data" \
        --volume "${dir}:/runs" \
        --workdir /runs \
        --log-driver local --log-opt max-size=100m --log-opt max-file=3 \
        "$IMAGE_NAME" \
        "$CONTAINER_TRAIN_BIN" "$@" >/dev/null
}

# wait_for_workers STAGE_LABEL
# Streams a status table every 30s (matching the original script), persists each
# worker's console log to its dir, removes the container, and dies if any worker
# exits non-zero or disappears. Uses the module-level WORKER_NAMES/WORKER_DIRS.
wait_for_workers() {
    local label="$1"
    local failure=0 running completed name state code i
    log "[${label}] workers active; status refreshes every 30s. Ctrl+C stops all workers cleanly."
    while :; do
        running=0
        completed=0
        for name in "${WORKER_NAMES[@]}"; do
            state="$(docker_cmd inspect --format '{{.State.Status}}' "$name" 2>/dev/null || echo removed)"
            case "$state" in
                running)
                    ((running += 1))
                    printf '  %-44s ' "$name"
                    docker_cmd logs --tail 1 "$name" 2>&1 || true
                    ;;
                exited)
                    code="$(docker_cmd inspect --format '{{.State.ExitCode}}' "$name")"
                    if [[ "$code" -eq 0 ]]; then ((completed += 1)); else failure=1; fi
                    ;;
                removed) failure=1 ;;
                *) failure=1 ;;
            esac
        done

        printf '\n[%s] %s workers_running=%d workers_finished=%d\n' "$(date -u +%FT%TZ)" "$label" "$running" "$completed"
        nvidia-smi --query-gpu=index,name,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu \
            --format=csv,noheader,nounits
        free -h | awk 'NR <= 2 {print}'

        if [[ "$failure" -ne 0 ]]; then
            for name in "${WORKER_NAMES[@]}"; do
                log "Last output from ${name}:"
                docker_cmd logs --tail 20 "$name" 2>&1 || true
            done
            for i in "${!WORKER_NAMES[@]}"; do
                docker_cmd logs "${WORKER_NAMES[$i]}" >"${WORKER_DIRS[$i]}/console.log" 2>&1 || true
            done
            die "[${label}] at least one training worker failed; stopping the remaining workers."
        fi
        [[ "$running" -eq 0 ]] && break
        sleep 30
    done

    for i in "${!WORKER_NAMES[@]}"; do
        docker_cmd logs "${WORKER_NAMES[$i]}" >"${WORKER_DIRS[$i]}/console.log" 2>&1 || true
        docker_cmd rm "${WORKER_NAMES[$i]}" >/dev/null || true
    done
    log "[${label}] all ${#WORKER_NAMES[@]} workers completed successfully."
}

# best_val_line FILE -> echoes the [VAL] line with the highest edge in FILE, or
# nothing if the file has no parseable [VAL] line. The [VAL] format is:
#   [VAL] loss=.. acc=.. dir_acc=.. | trades=N winrate=.. edge=.. pf=.. kelly=.. | ..
best_val_line() {
    local file="$1"
    [[ -r "$file" ]] || return 0
    awk '
        { gsub(/\033\[[0-9;]*m/, "", $0) }
        /\[VAL\]/ {
            edge = ""
            n = split($0, toks, /[[:space:]]+/)
            for (i = 1; i <= n; i++) {
                if (toks[i] ~ /^edge=/) { split(toks[i], kv, "="); edge = kv[2] + 0; has = 1 }
            }
            if (has && (best_set == 0 || edge > best_edge)) { best_edge = edge; best = $0; best_set = 1 }
            has = 0
        }
        END { if (best_set) print best }
    ' "$file"
}

# parse_val_field LINE KEY -> echoes the numeric value of KEY= in LINE (or empty).
parse_val_field() {
    local line="$1" key="$2"
    awk -v key="$key" '
        {
            gsub(/\033\[[0-9;]*m/, "", $0)
            n = split($0, toks, /[[:space:]]+/)
            for (i = 1; i <= n; i++) {
                if (toks[i] ~ "^" key "=") { split(toks[i], kv, "="); print kv[2]; exit }
            }
        }
    ' <<<"$line"
}

# cache_key STRIDE HORIZON THRESHOLD MAX_SAMPLES -> the derived cache file name.
# Window is fixed at 512 (matches the existing pipeline / phase-2 key format).
cache_key() {
    local stride="$1" horizon="$2" threshold="$3" max_samples="$4"
    # Threshold is formatted as fixed 2-decimals (t0.50, t1.00, ...) to match the
    # binary's cache filename exactly; stride/horizon/n stay integers (BUG 2).
    printf 'data.w512_s%d_h%d_t%.2f_n%d.cache' "$stride" "$horizon" "$threshold" "$max_samples"
}

# ============================================================================
# STAGE 0 - CANARY GATE (~2 min, GPU 0 only).
# Run the FIRST sweep config briefly and abort the whole launcher unless the
# binary trains cleanly. This protects the ~6h GPU window from a broken build.
# ============================================================================
read -r CANARY_THRESHOLD CANARY_HORIZON CANARY_STRIDE CANARY_LR _canary_ms <<<"${SWEEP_GRID[0]}"
CANARY_DIR="${RUN_ROOT}/canary"
CANARY_LOG="${CANARY_DIR}/canary.log"
mkdir -p "$CANARY_DIR"

log "STAGE 0 canary gate: thr=${CANARY_THRESHOLD} horizon=${CANARY_HORIZON} stride=${CANARY_STRIDE} lr=${CANARY_LR} on GPU 0 (~2 min)."
canary_rc=0
docker_cmd run --rm \
    --name "from-canary-$$" \
    --gpus 'device=0' \
    --cpuset-cpus '0-11' \
    --memory "$WORKER_MEMORY" \
    --memory-swap "$WORKER_MEMORY" \
    --shm-size 8g \
    --ulimit memlock=-1:-1 \
    --env OMP_NUM_THREADS="$CPU_THREADS_PER_GPU" \
    --env OMP_PROC_BIND=close \
    --env OMP_PLACES=cores \
    --env MALLOC_ARENA_MAX=4 \
    --volume "${ROOT_DIR}:/workspace:ro" \
    --volume "${DATA_DIR}:/data" \
    --volume "${CANARY_DIR}:/runs" \
    --workdir /runs \
    "$IMAGE_NAME" \
    "$CONTAINER_TRAIN_BIN" train --data "/data/${DATA_FILE}" \
        --seed 42 --output-prefix "/runs/canary" \
        --lr "$CANARY_LR" --batch-size "$SWEEP_BATCH_SIZE" \
        --direction-threshold "$CANARY_THRESHOLD" \
        --horizon "$CANARY_HORIZON" --stride "$CANARY_STRIDE" \
        --max-steps "$CANARY_MAX_STEPS" --max-samples "$CANARY_MAX_SAMPLES" \
        --validate-every "$CANARY_VALIDATE_EVERY" 2>&1 | tee "$CANARY_LOG" || canary_rc="${PIPESTATUS[0]}"

[[ "$canary_rc" -eq 0 ]] || die "STAGE 0 canary aborted: train binary exited ${canary_rc}. See ${CANARY_LOG}."

# (a) no CUDA/cuBLAS faults in the output.
if grep -Eq 'cuBLAS error|CUDA error|illegal value' "$CANARY_LOG"; then
    grep -En 'cuBLAS error|CUDA error|illegal value' "$CANARY_LOG" | head -5 >&2
    die "STAGE 0 canary aborted: GPU error string found in ${CANARY_LOG}."
fi

# (b) it reached step>=3000 and printed a [DONE] or [VAL] line.
canary_max_step="$(grep -Eo 'step[ =]+[0-9]+' "$CANARY_LOG" | grep -Eo '[0-9]+' | sort -n | tail -1 || true)"
canary_max_step="${canary_max_step:-0}"
if ! grep -Eq '\[DONE\]|\[VAL\]' "$CANARY_LOG"; then
    die "STAGE 0 canary aborted: no [DONE]/[VAL] line in ${CANARY_LOG}."
fi
if [[ "$canary_max_step" -lt "$CANARY_MAX_STEPS" ]]; then
    die "STAGE 0 canary aborted: reached step ${canary_max_step} < ${CANARY_MAX_STEPS}. See ${CANARY_LOG}."
fi

# (c) at least one [VAL] line has a finite edge, and EXPLODE was not a flood.
canary_explosions="$(grep -c '\[EXPLODE\]' "$CANARY_LOG" || true)"
canary_explosions="${canary_explosions:-0}"
if [[ "$canary_explosions" -gt "$CANARY_EXPLODE_LIMIT" ]]; then
    die "STAGE 0 canary aborted: ${canary_explosions} [EXPLODE] lines (> ${CANARY_EXPLODE_LIMIT}). Training is diverging. See ${CANARY_LOG}."
fi
canary_best="$(best_val_line "$CANARY_LOG")"
[[ -n "$canary_best" ]] || die "STAGE 0 canary aborted: no [VAL] line with a parseable edge in ${CANARY_LOG}."
canary_edge="$(parse_val_field "$canary_best" edge)"
# Strip any residual ANSI color escapes before testing the number (BUG 1).
canary_edge="$(printf '%s' "$canary_edge" | sed $'s/\x1b\[[0-9;]*m//g')"
case "$canary_edge" in
    ''|*[!0-9eE.+-]*) die "STAGE 0 canary aborted: edge='${canary_edge}' is not a finite number. See ${CANARY_LOG}." ;;
esac
if ! awk -v e="$canary_edge" 'BEGIN { if (e+0==e && e==e && e!="inf" && e!="-inf" && e!="nan") exit 0; exit 1 }' </dev/null; then
    die "STAGE 0 canary aborted: edge='${canary_edge}' is not finite. See ${CANARY_LOG}."
fi
log "STAGE 0 canary PASSED: reached step ${canary_max_step}, ${canary_explosions} explode lines, best edge=${canary_edge}."

# ============================================================================
# STAGE 1 - CONFIG SWEEP (8 GPUs in parallel, ~75 min).
# One worker per GPU, each with its own (threshold,horizon,stride,lr,max_samples)
# from SWEEP_GRID. Each distinct config derives + builds its own cache on first
# use, so configs never collide.
# ============================================================================
SWEEP_ROOT="${RUN_ROOT}/sweep"
mkdir -p "$SWEEP_ROOT"
WORKER_NAMES=()
WORKER_DIRS=()
SWEEP_GPU_OF_INDEX=()

log "STAGE 1 config sweep: launching ${GPU_COUNT} workers (1 L4 + 12 pinned vCPUs + 42g each)."
for gpu in $(seq 0 7); do
    read -r s_threshold s_horizon s_stride s_lr s_max_samples <<<"${SWEEP_GRID[$gpu]}"
    cpu_first=$((gpu * CPU_THREADS_PER_GPU))
    cpu_last=$((cpu_first + CPU_THREADS_PER_GPU - 1))
    worker_dir="${SWEEP_ROOT}/gpu-${gpu}"
    worker_name="from-sweep-${USER:-user}-${gpu}-$$"
    WORKER_NAMES+=("$worker_name")
    WORKER_DIRS+=("$worker_dir")
    SWEEP_GPU_OF_INDEX+=("$gpu")

    log "  gpu${gpu}: thr=${s_threshold} horizon=${s_horizon} stride=${s_stride} lr=${s_lr} max_samples=${s_max_samples} cpus=${cpu_first}-${cpu_last}"
    launch_worker "$gpu" "$cpu_first" "$cpu_last" "$worker_name" "$worker_dir" -- \
        train --data "/data/${DATA_FILE}" \
            --seed 42 --output-prefix "/runs/sweep_gpu${gpu}" \
            --lr "$s_lr" --batch-size "$SWEEP_BATCH_SIZE" \
            --direction-threshold "$s_threshold" \
            --horizon "$s_horizon" --stride "$s_stride" \
            --max-steps "$SWEEP_MAX_STEPS" --max-samples "$s_max_samples" \
            --validate-every "$SWEEP_VALIDATE_EVERY" --save-every "$SWEEP_SAVE_EVERY"
done

cat >"${RUN_ROOT}/run-config.txt" <<EOF
utc_started=$(date -u +%FT%TZ)
host=$(hostname)
data=${DATA_PATH}
stage1_batch_size=${SWEEP_BATCH_SIZE}
stage1_max_steps=${SWEEP_MAX_STEPS}
stage1_validate_every=${SWEEP_VALIDATE_EVERY}
stage1_save_every=${SWEEP_SAVE_EVERY}
stage1_grid=$(printf '%s; ' "${SWEEP_GRID[@]}")
stage2_max_samples=${ENSEMBLE_MAX_SAMPLES}
stage2_max_steps=${ENSEMBLE_MAX_STEPS}
ensemble_seeds=${ENSEMBLE_SEEDS[*]}
workers=${WORKER_NAMES[*]}
EOF

wait_for_workers "STAGE 1 sweep"

# ----------------------------------------------------------------------------
# SELECTION: parse each worker's console log for its best [VAL] line, build a
# leaderboard sorted by edge desc, and pick the winner = highest edge among
# configs with kelly>0 and trades>=200. If none qualify, pick highest edge
# overall and warn loudly that no config showed positive Kelly.
# ----------------------------------------------------------------------------
LEADERBOARD="${RUN_ROOT}/leaderboard.txt"
{
    printf '# Stage 1 config sweep leaderboard (best [VAL] edge per config, sorted desc)\n'
    printf '# generated %s\n' "$(date -u +%FT%TZ)"
    printf '#\n'
    printf '# %-4s %-9s %-7s %-6s %-8s %-12s %-9s %-8s %-9s %-9s %-9s\n' \
        gpu thr horizon stride lr max_samp edge kelly pf winrate trades
} >"$LEADERBOARD"

# Collect one record per GPU: "edge gpu thr horizon stride lr max_samples kelly pf winrate trades"
SWEEP_RECORDS=()
for idx in "${!WORKER_DIRS[@]}"; do
    gpu="${SWEEP_GPU_OF_INDEX[$idx]}"
    read -r r_threshold r_horizon r_stride r_lr r_max_samples <<<"${SWEEP_GRID[$gpu]}"
    log_file="${WORKER_DIRS[$idx]}/console.log"
    best="$(best_val_line "$log_file")"
    if [[ -z "$best" ]]; then
        log "WARNING: no parseable [VAL] line for gpu${gpu} (${log_file}); excluding from selection."
        edge="nan"; kelly="nan"; pf="nan"; winrate="nan"; trades="0"
    else
        edge="$(parse_val_field "$best" edge)"
        kelly="$(parse_val_field "$best" kelly)"
        pf="$(parse_val_field "$best" pf)"
        winrate="$(parse_val_field "$best" winrate)"
        trades="$(parse_val_field "$best" trades)"
        edge="${edge:-nan}"; kelly="${kelly:-nan}"; pf="${pf:-nan}"; winrate="${winrate:-nan}"; trades="${trades:-0}"
    fi
    SWEEP_RECORDS+=("${edge}|${gpu}|${r_threshold}|${r_horizon}|${r_stride}|${r_lr}|${r_max_samples}|${kelly}|${pf}|${winrate}|${trades}")
done

# Sort records by edge desc (numeric; non-numeric sinks to the bottom).
mapfile -t SWEEP_SORTED < <(
    for rec in "${SWEEP_RECORDS[@]}"; do
        e="${rec%%|*}"
        case "$e" in
            ''|nan|inf|-inf|*[!0-9eE.+-]*) key="-1e308" ;;
            *) key="$e" ;;
        esac
        printf '%s\t%s\n' "$key" "$rec"
    done | sort -t$'\t' -k1,1 -g -r | cut -f2-
)

for rec in "${SWEEP_SORTED[@]}"; do
    IFS='|' read -r e gpu thr horizon stride lr max_samples kelly pf winrate trades <<<"$rec"
    printf '  %-4s %-9s %-7s %-6s %-8s %-12s %-9s %-8s %-9s %-9s %-9s\n' \
        "$gpu" "$thr" "$horizon" "$stride" "$lr" "$max_samples" "$e" "$kelly" "$pf" "$winrate" "$trades" \
        >>"$LEADERBOARD"
done

# Winner selection: highest edge among kelly>0 and trades>=200.
WIN_REC=""
for rec in "${SWEEP_SORTED[@]}"; do
    IFS='|' read -r e gpu thr horizon stride lr max_samples kelly pf winrate trades <<<"$rec"
    if awk -v k="$kelly" -v t="$trades" 'BEGIN { if (k+0>0 && t+0>=200) exit 0; exit 1 }'; then
        WIN_REC="$rec"
        break
    fi
done

if [[ -n "$WIN_REC" ]]; then
    SWEEP_WINNER_QUALIFIED=1
else
    # Fall back to highest edge overall; warn that no config showed positive Kelly.
    SWEEP_WINNER_QUALIFIED=0
    WIN_REC="${SWEEP_SORTED[0]:-}"
    [[ -n "$WIN_REC" ]] || die "SELECTION failed: no sweep records to choose a winner from."
    log "WARNING: no sweep config showed positive Kelly with trades>=200."
    log "WARNING: falling back to the highest-edge config; treat the ensemble as exploratory, not shippable."
fi

IFS='|' read -r WIN_EDGE WIN_GPU WIN_THRESHOLD WIN_HORIZON WIN_STRIDE WIN_LR WIN_MAX_SAMPLES WIN_KELLY WIN_PF WIN_WINRATE WIN_TRADES <<<"$WIN_REC"

{
    printf '\n# WINNING CONFIG (qualified_kelly_positive=%s)\n' "$SWEEP_WINNER_QUALIFIED"
    printf '# gpu=%s direction_threshold=%s horizon=%s stride=%s lr=%s\n' \
        "$WIN_GPU" "$WIN_THRESHOLD" "$WIN_HORIZON" "$WIN_STRIDE" "$WIN_LR"
    printf '# best edge=%s kelly=%s pf=%s winrate=%s trades=%s\n' \
        "$WIN_EDGE" "$WIN_KELLY" "$WIN_PF" "$WIN_WINRATE" "$WIN_TRADES"
} >>"$LEADERBOARD"

log "SELECTION winner: gpu${WIN_GPU} thr=${WIN_THRESHOLD} horizon=${WIN_HORIZON} stride=${WIN_STRIDE} lr=${WIN_LR} (edge=${WIN_EDGE} kelly=${WIN_KELLY} pf=${WIN_PF} trades=${WIN_TRADES})."
log "Leaderboard written to ${LEADERBOARD}."

# ============================================================================
# STAGE 2 - ENSEMBLE WINNER (8 GPUs, remaining time).
# Relaunch 8 workers with the winning (threshold,horizon,stride,lr) but 8
# different seeds, under the phase-2 budget. Pre-warm the winner's cache file
# (key derived from w512,stride,horizon,threshold,max_samples) once on GPU 0 so
# the 8 seeded runs all load instantly instead of each reprocessing the parquet.
# ============================================================================
ENSEMBLE_ROOT="${RUN_ROOT}/ensemble"
mkdir -p "$ENSEMBLE_ROOT"
WIN_CACHE_KEY="$(cache_key "$WIN_STRIDE" "$WIN_HORIZON" "$WIN_THRESHOLD" "$ENSEMBLE_MAX_SAMPLES")"
log "STAGE 2 ensemble: pre-warming winner cache ${WIN_CACHE_KEY} (max_samples=${ENSEMBLE_MAX_SAMPLES}) on GPU 0."
docker_cmd run --rm \
    --name "from-cachewarm-$$" \
    --gpus 'device=0' \
    --cpuset-cpus '0-95' \
    --memory 300g \
    --memory-swap 300g \
    --shm-size 8g \
    --ulimit memlock=-1:-1 \
    --env OMP_NUM_THREADS=96 \
    --env OMP_PROC_BIND=spread \
    --env OMP_PLACES=cores \
    --volume "${ROOT_DIR}:/workspace:ro" \
    --volume "${DATA_DIR}:/data" \
    --volume "${ENSEMBLE_ROOT}:/runs" \
    --workdir /runs \
    "$IMAGE_NAME" \
    "$CONTAINER_TRAIN_BIN" train --data "/data/${DATA_FILE}" \
        --cache-only \
        --direction-threshold "$WIN_THRESHOLD" \
        --horizon "$WIN_HORIZON" --stride "$WIN_STRIDE" \
        --max-samples "$ENSEMBLE_MAX_SAMPLES" --chunk-size 50000000

WORKER_NAMES=()
WORKER_DIRS=()
ENSEMBLE_SEED_OF_INDEX=()

log "STAGE 2 ensemble: launching ${GPU_COUNT} seeded workers of the winning config."
for gpu in $(seq 0 7); do
    seed="${ENSEMBLE_SEEDS[$gpu]}"
    cpu_first=$((gpu * CPU_THREADS_PER_GPU))
    cpu_last=$((cpu_first + CPU_THREADS_PER_GPU - 1))
    worker_dir="${ENSEMBLE_ROOT}/gpu-${gpu}-seed-${seed}"
    worker_name="from-ens-${USER:-user}-${gpu}-${seed}-$$"
    WORKER_NAMES+=("$worker_name")
    WORKER_DIRS+=("$worker_dir")
    ENSEMBLE_SEED_OF_INDEX+=("$seed")

    log "  gpu${gpu}: seed=${seed} thr=${WIN_THRESHOLD} horizon=${WIN_HORIZON} stride=${WIN_STRIDE} lr=${WIN_LR} max_samples=${ENSEMBLE_MAX_SAMPLES} cpus=${cpu_first}-${cpu_last}"
    launch_worker "$gpu" "$cpu_first" "$cpu_last" "$worker_name" "$worker_dir" -- \
        train --data "/data/${DATA_FILE}" \
            --seed "$seed" --output-prefix "/runs/weights_seed${seed}" \
            --lr "$WIN_LR" --batch-size "$ENSEMBLE_BATCH_SIZE" \
            --direction-threshold "$WIN_THRESHOLD" \
            --horizon "$WIN_HORIZON" --stride "$WIN_STRIDE" \
            --max-steps "$ENSEMBLE_MAX_STEPS" --max-samples "$ENSEMBLE_MAX_SAMPLES" \
            --validate-every "$ENSEMBLE_VALIDATE_EVERY" --save-every "$ENSEMBLE_SAVE_EVERY"
done

wait_for_workers "STAGE 2 ensemble"

# Append each seed's best edge/kelly to the leaderboard.
{
    printf '\n# Stage 2 ensemble of winning config (gpu%s: thr=%s horizon=%s stride=%s lr=%s, cache=%s)\n' \
        "$WIN_GPU" "$WIN_THRESHOLD" "$WIN_HORIZON" "$WIN_STRIDE" "$WIN_LR" "$WIN_CACHE_KEY"
    printf '# %-8s %-12s %-12s\n' seed edge kelly
} >>"$LEADERBOARD"

for idx in "${!WORKER_DIRS[@]}"; do
    seed="${ENSEMBLE_SEED_OF_INDEX[$idx]}"
    best="$(best_val_line "${WORKER_DIRS[$idx]}/console.log")"
    if [[ -z "$best" ]]; then
        log "WARNING: no parseable [VAL] line for ensemble seed ${seed}."
        e_edge="nan"; e_kelly="nan"
    else
        e_edge="$(parse_val_field "$best" edge)"; e_edge="${e_edge:-nan}"
        e_kelly="$(parse_val_field "$best" kelly)"; e_kelly="${e_kelly:-nan}"
    fi
    printf '  %-8s %-12s %-12s\n' "$seed" "$e_edge" "$e_kelly" >>"$LEADERBOARD"
done

trap - INT TERM EXIT
log "All three stages completed. Winner: gpu${WIN_GPU} thr=${WIN_THRESHOLD} horizon=${WIN_HORIZON} stride=${WIN_STRIDE} lr=${WIN_LR}."
log "Leaderboard: ${LEADERBOARD}"
log "Outputs: ${RUN_ROOT}"
