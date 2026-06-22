#!/usr/bin/env bash
# Adversarial NULL / overfitting stress engine.
# 1) generate N synthetic NULL tick series (no real edge): half GBM, half block-bootstrap
#    of the real 2023 returns. 2) run ONE representative walk-forward config on each across
#    8 GPUs. 3) summarize the empirical NULL distribution of holdout PF / edge / t.
# A REAL model whose holdout stat exceeds the 95th/99th null percentile is genuinely
# significant (deflated). Finishes in ~12 min — well before the dense fetch.
set -uo pipefail
N="${N:-48}"
SRC="${SRC:-/data/XAUUSD_ticks_2023.parquet}"           # for bootstrap method (container path)
NCT="${NCT:-2000000}"
CMD="${CMD:-wfdeep}"                                     # representative arch
HORIZON="${HORIZON:-8192}"; COSTM="${COSTM:-1.5}"; GATE="${GATE:-0.55}"; WIN="${WIN:-256}"; BATCH="${BATCH:-1024}"
IMAGE="${IMAGE:-from-g2-l4:cuda12.8}"; BIN="/workspace/build-l4/from"; NGPUS=8
NDIR="$HOME/data/null"; RUNS="$HOME/runs/null"; mkdir -p "$NDIR" "$RUNS"
SRC_HOST="$HOME/data/$(basename "$SRC")"

echo "[null] generating $N null datasets ($NCT ticks each) ..."
gen() {
  i="$1"; pad="$(printf '%03d' "$i")"
  if [ $((i % 2)) -eq 0 ]; then
    python3 ~/gen_null.py --out "$NDIR/null_${pad}.parquet" --seed "$i" --method gbm --n-ticks "$NCT" >/dev/null 2>&1
  else
    python3 ~/gen_null.py --out "$NDIR/null_${pad}.parquet" --seed "$i" --method bootstrap --src "$SRC_HOST" --n-ticks "$NCT" >/dev/null 2>&1
  fi
}
export -f gen; export NDIR NCT SRC_HOST
seq 0 $((N-1)) | xargs -P 32 -I{} bash -c 'gen "$@"' _ {}
echo "[null] generated $(ls "$NDIR"/null_*.parquet 2>/dev/null | wc -l) datasets"

# 8-GPU queue over the null datasets.
declare -a CUR; for g in $(seq 0 7); do CUR[$g]=""; done
files=("$NDIR"/null_*.parquet); NEXT=0; TOTAL=${#files[@]}; done_n=0
launch() { local g="$1" f="$2" b; b="$(basename "$f" .parquet)"
  sudo docker rm -f "ns_${b}" >/dev/null 2>&1 || true
  sudo docker run -d --name "ns_${b}" --gpus "device=${g}" --cpuset-cpus "$((g*12))-$((g*12+11))" \
    --memory 42g --shm-size 8g --user "$(id -u):$(id -g)" -e OMP_NUM_THREADS=12 \
    -v "$HOME/from:/workspace" -v "$NDIR:/null" -v "$HOME/data:/data" -v "$RUNS:/runs" -w /workspace \
    "$IMAGE" "$BIN" "$CMD" --data "/null/$(basename "$f")" --window "$WIN" --horizon "$HORIZON" \
    --cost-mult "$COSTM" --barrier-k 1.0 --conf-gate "$GATE" --stride 64 --max-samples "$NCT" \
    --folds 6 --holdout-frac 0.15 --batch-size "$BATCH" --lr 0.0005 --max-steps 2000 --epochs 8 \
    --model-dir /runs/null_reg --output "/runs/${b}.txt" >/dev/null 2>&1
}
for g in $(seq 0 7); do [ "$NEXT" -lt "$TOTAL" ] && { launch "$g" "${files[$NEXT]}"; CUR[$g]="ns_$(basename "${files[$NEXT]}" .parquet)"; NEXT=$((NEXT+1)); }; done
while :; do running=0
  for g in $(seq 0 7); do nm="${CUR[$g]}"; [ -z "$nm" ] && continue
    st="$(sudo docker inspect -f '{{.State.Status}}' "$nm" 2>/dev/null || echo gone)"
    if [ "$st" = running ]; then running=$((running+1)); else
      sudo docker rm "$nm" >/dev/null 2>&1 || true; done_n=$((done_n+1)); CUR[$g]=""
      if [ "$NEXT" -lt "$TOTAL" ]; then launch "$g" "${files[$NEXT]}"; CUR[$g]="ns_$(basename "${files[$NEXT]}" .parquet)"; NEXT=$((NEXT+1)); running=$((running+1)); fi
    fi
  done
  echo "--- $(date -u +%H:%M:%S) null running=$running done=$done_n/$TOTAL ---"
  [ "$running" -eq 0 ] && [ "$NEXT" -ge "$TOTAL" ] && break; sleep 15
done

# Collect + percentiles.
echo "file,pf,edge,t,neff" > "$RUNS/null_results.csv"
for f in "$RUNS"/null_*.txt; do
  l="$(grep -h '^EDGE:' "$f" 2>/dev/null | tail -1)"
  pf="$(printf '%s' "$l" | grep -oE 'PF=[-0-9.]+' | cut -d= -f2)"
  kel="$(printf '%s' "$l" | grep -oE 'kelly=[-0-9.]+' | cut -d= -f2)"
  tt="$(printf '%s' "$l" | grep -oE 't=[-0-9.]+' | cut -d= -f2)"
  nf="$(printf '%s' "$l" | grep -oE 'N_eff=[-0-9.]+' | cut -d= -f2)"
  [ -n "$pf" ] && echo "$(basename "$f"),$pf,${kel:-0},${tt:-0},${nf:-0}" >> "$RUNS/null_results.csv"
done
python3 - "$RUNS/null_results.csv" <<'PY'
import csv,sys,statistics as st
rows=list(csv.DictReader(open(sys.argv[1])))
for col in ("pf","t"):
    v=sorted(float(r[col]) for r in rows if r.get(col))
    if not v: continue
    q=lambda p: v[min(len(v)-1,int(p*len(v)))]
    print(f"NULL {col}: n={len(v)} mean={sum(v)/len(v):.3f} p50={q(.5):.3f} p90={q(.9):.3f} p95={q(.95):.3f} p99={q(.99):.3f} max={v[-1]:.3f}")
print("=> a REAL model is significant only if its holdout stat exceeds the p95/p99 NULL value above.")
PY
echo "[null] done. distribution -> $RUNS/null_results.csv"
