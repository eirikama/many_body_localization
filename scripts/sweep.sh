#!/usr/bin/env bash
set -euo pipefail

# ── Parameters ────────────────────────────────────────────────────────────────
L_values=(6 8 10 12)
W_start=3.1
W_end=12.0
W_step=0.5
dup_start=1
dup_end=10

BINARY=./build/solve_random_heisenberg

# ── Sweep ─────────────────────────────────────────────────────────────────────
echo "Starting sweep"
echo "L values  : ${L_values[*]}"
echo "W range   : ${W_start} → ${W_end} (step ${W_step})"
echo "Duplicates: ${dup_start} → ${dup_end}"
echo "────────────────────────────────────────"

start_time=$SECONDS

for L in "${L_values[@]}"; do
    W=$W_start
    while (( $(echo "$W <= $W_end" | bc -l) )); do
        for (( dupli=dup_start; dupli<=dup_end; dupli++ )); do
            printf "L=%-3s  W=%-6s  dupli=%d\n" "$L" "$W" "$dupli"
            "$BINARY" "$L" "$W" "$dupli"
        done
        W=$(echo "scale=6; $W + $W_step" | bc -l)
    done
done

elapsed=$(( SECONDS - start_time ))
echo "────────────────────────────────────────"
printf "Done in %dm %ds.\n" "$(( elapsed/60 ))" "$(( elapsed%60 ))"