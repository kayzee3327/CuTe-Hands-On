#!/usr/bin/env bash
set -u
cd /home/kz/TsingHua-PACMAN/UnifiedIR/02-cudagemm/CuTe-Hands-On
SRC=src/sgemm_opt86.cu
OUT=/tmp/sweep_results.csv
RAW=/tmp/sweep_raw
mkdir -p "$RAW"
echo "bP,KREG,status,regs,occ_pct,issue_pct,warpcyc_per_inst,stall_wait,stall_dispatch,stall_barrier,stall_mio,stall_longsb,stall_shortsb,cycles" > "$OUT"

METRICS=sm__cycles_elapsed.avg,launch__registers_per_thread,sm__warps_active.avg.pct_of_peak_sustained_active,smsp__issue_active.avg.pct_of_peak_sustained_elapsed,smsp__average_warp_latency_per_inst_issued.ratio,smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio,smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio,smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio,smsp__average_warps_issue_stalled_wait_per_issue_active.ratio,smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio,smsp__average_warps_issue_stalled_dispatch_stall_per_issue_active.ratio

for bp in 2 3 4 5; do
  for kreg in 2 3 4; do
    echo "==== bP=$bp KREG=$kreg ===="
    sed -i -E "s/^#define SGEMM_BP [0-9]+/#define SGEMM_BP $bp/" "$SRC"
    sed -i -E "s/^#define SGEMM_KREG [0-9]+/#define SGEMM_KREG $kreg/" "$SRC"
    if ! cmake --build build --target sgemm_opt86 >/tmp/build_${bp}_${kreg}.log 2>&1; then
      echo "$bp,$kreg,BUILD_FAIL,,,,,,,,,,," >> "$OUT"; continue
    fi
    CSV=$RAW/r_${bp}_${kreg}.csv
    # NCU run; program stdout (incl. correctness "Status: PASSED/FAILED") + CSV land in $CSV
    ncu --target-processes all -k "regex:sgemm_opt86_nt_v4" -c 1 --csv \
        --metrics "$METRICS" ./build/sgemm_opt86 -p >"$CSV" 2>/tmp/ncu_${bp}_${kreg}.log
    if grep -q "Status           : PASSED" "$CSV"; then ST=PASS
    elif grep -q "Status           : FAILED" "$CSV"; then ST=FAILED
    else ST=UNKNOWN; fi
    python3 - "$CSV" "$bp" "$kreg" "$ST" >> "$OUT" <<'PY'
import csv,sys
path,bp,kreg,st=sys.argv[1:5]
m={}
try:
  with open(path) as f:
    for row in csv.reader(f):
      if len(row)>=15 and row[0]=="0":
        m[row[-3]]=row[-1]   # -3=Metric Name, -2=Unit, -1=Value
except Exception:
  pass
def g(k): return m.get(k,"")
print(",".join([bp,kreg,st,
  g("launch__registers_per_thread"),
  g("sm__warps_active.avg.pct_of_peak_sustained_active"),
  g("smsp__issue_active.avg.pct_of_peak_sustained_elapsed"),
  g("smsp__average_warp_latency_per_inst_issued.ratio"),
  g("smsp__average_warps_issue_stalled_wait_per_issue_active.ratio"),
  g("smsp__average_warps_issue_stalled_dispatch_stall_per_issue_active.ratio"),
  g("smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio"),
  g("smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio"),
  g("smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio"),
  g("smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio"),
  g("sm__cycles_elapsed.avg")]))
PY
  done
done
# restore defaults
sed -i -E "s/^#define SGEMM_BP [0-9]+/#define SGEMM_BP 2/" "$SRC"
sed -i -E "s/^#define SGEMM_KREG [0-9]+/#define SGEMM_KREG 2/" "$SRC"
echo "DONE"
