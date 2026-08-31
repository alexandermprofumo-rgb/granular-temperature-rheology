#!/bin/bash
# 3D frictionless test: does n fall below 1/6 at mu_g=0, the way it fell
# below 1/8 in 2D? mu_g=0.3 and 1.0 are included as finite-friction
# reference points (expect these closer to 1/6, per Irmer et al.-like
# frictional 3D flows).

set -e

if command -v timeout >/dev/null 2>&1; then
  TIMEOUT_CMD="timeout 1800"
elif command -v gtimeout >/dev/null 2>&1; then
  TIMEOUT_CMD="gtimeout 1800"
else
  echo "WARNING: no 'timeout'/'gtimeout' found -- runs won't be time-capped."
  TIMEOUT_CMD=""
fi

MU_G_VALUES=(0.0 0.05 0.1 0.15 0.2 0.3 0.5 1.0)
TGRAN_VALUES=(0.0005 0.001 0.003 0.008 0.02)
SEEDS=(1 2 3)

N=4000
PHI=0.64      # calibrated: gives Z~6 (frictionless isostatic) at Pconf below
PCONF=10.0    # calibrated: gives Z~6 at phi=0.64 (see calibration notes)
GDOT=1.0e-3
NEQUIL=40000
NMEAS=80000

mkdir -p sweep_run_3d
cd sweep_run_3d

for seed in "${SEEDS[@]}"; do
  datafile="data.granular3d_seed${seed}"
  if [ ! -f "$datafile" ]; then
    python3 ../gen_data_3d.py --N ${N} --phi ${PHI} --seed ${seed} --out "$datafile"
  fi

  for mu_g in "${MU_G_VALUES[@]}"; do
    for Tgran in "${TGRAN_VALUES[@]}"; do
      label="mu${mu_g}_T${Tgran}_s${seed}"
      if [ -f "log.${label}" ] && grep -q "^DONE" "log.${label}"; then
        echo "skip (already done): ${label}"
        continue
      fi
      echo "=== running ${label} ==="
      set +e
      ${TIMEOUT_CMD} lmp_serial -in ../in.granular_3d \
          -var datafile "$datafile" \
          -var mu_g ${mu_g} \
          -var gdot ${GDOT} \
          -var Pconf ${PCONF} \
          -var Tgran ${Tgran} \
          -var seed ${seed} \
          -var label "$label" \
          -var nequil ${NEQUIL} \
          -var nmeas ${NMEAS} \
          -log none > "run.${label}.out" 2>&1
      status=$?
      set -e
      if [ $status -eq 124 ]; then
        echo "${label}" >> ../timed_out_runs_3d.txt
        echo "  !! TIMED OUT after 30 min, logged, continuing"
      fi
    done
  done
done

python3 ../analyze_sweep_3d.py --glob "log.*" --out ../results_3d.csv
