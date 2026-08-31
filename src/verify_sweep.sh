#!/bin/bash
# Independent verification re-run: fresh seed (101, never used in the
# original campaign), same production parameters (N=4000, core Theta
# window), 4 mu_g values spanning the reported curve shape.
set -e
if command -v timeout >/dev/null 2>&1; then TIMEOUT_CMD="timeout 1800"
elif command -v gtimeout >/dev/null 2>&1; then TIMEOUT_CMD="gtimeout 1800"
else TIMEOUT_CMD=""; fi

MU_G_VALUES=(0.0 0.1 0.3 1.0)
TGRAN_VALUES=(0.0005 0.002 0.006 0.015)
SEED=101
N=4000
PHI=0.80
GDOT=1.0e-3
PCONF=2.0
NEQUIL=40000
NMEAS=80000

cd verification_run
datafile="data.granular_seed${SEED}"
if [ ! -f "$datafile" ]; then
  python3 ../gen_data.py --N ${N} --phi ${PHI} --seed ${SEED} --out "$datafile"
fi

for mu_g in "${MU_G_VALUES[@]}"; do
  for Tgran in "${TGRAN_VALUES[@]}"; do
    label="mu${mu_g}_T${Tgran}_s${SEED}"
    if [ -f "log.${label}" ] && grep -q "^DONE" "log.${label}"; then
      echo "skip (already done): ${label}"; continue
    fi
    echo "=== running ${label} ==="
    ${TIMEOUT_CMD} lmp_serial -in ../in.granular_2d \
        -var datafile "$datafile" -var mu_g ${mu_g} -var gdot ${GDOT} \
        -var Pconf ${PCONF} -var Tgran ${Tgran} -var seed ${SEED} \
        -var label "$label" -var nequil ${NEQUIL} -var nmeas ${NMEAS} \
        -log none > "run.${label}.out" 2>&1
    echo "  done: ${label}"
  done
done

python3 ../analyze_sweep.py --glob "log.*" --out ../verification_results.csv
