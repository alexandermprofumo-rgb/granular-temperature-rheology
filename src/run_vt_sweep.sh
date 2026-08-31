#!/bin/bash
# vt-scaling sweep: for each mu_g on the macroscopic n(mu_g) curve, measure
# how the typical relative tangential (sliding) velocity at a contact
# scales with Theta. Compares against the macroscopic exponent to test
# whether the microscopic sliding-velocity response actually explains the
# rise/peak/fall shape found in mu(Theta).

set -e

if command -v timeout >/dev/null 2>&1; then
  TIMEOUT_CMD="timeout 1200"
elif command -v gtimeout >/dev/null 2>&1; then
  TIMEOUT_CMD="gtimeout 1200"
else
  echo "WARNING: no 'timeout'/'gtimeout' found -- runs won't be time-capped."
  TIMEOUT_CMD=""
fi

# mirrors key points on the already-measured 2D n(mu_g) curve: rising
# branch (0, 0.05, 0.1), peak region (0.3), falling branch (1.0)
MU_G_VALUES=(0.0 0.05 0.1 0.3 1.0)
TGRAN_VALUES=(0.0005 0.002 0.006 0.015)
SEEDS=(1 2 3)

N=2000
PHI=0.80
PCONF=2.0
GDOT=1.0e-3
NEQUIL=40000     # same equilibration length as the main mu(Theta) sweep,
                 # so this measures the same physical steady state
NTRACK=3000
DUMP_EVERY=15

mkdir -p sweep_vt
cd sweep_vt

for seed in "${SEEDS[@]}"; do
  datafile="data.granular_vt_seed${seed}"
  if [ ! -f "$datafile" ]; then
    python3 ../gen_data.py --N ${N} --phi ${PHI} --seed ${seed} --out "$datafile"
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
      ${TIMEOUT_CMD} lmp_serial -in ../in.contact_tracking_2d \
          -var datafile "$datafile" \
          -var mu_g ${mu_g} \
          -var gdot ${GDOT} \
          -var Pconf ${PCONF} \
          -var Tgran ${Tgran} \
          -var seed ${seed} \
          -var label "$label" \
          -var nequil ${NEQUIL} \
          -var ntrack ${NTRACK} \
          -var dump_every ${DUMP_EVERY} \
          > "run.${label}.out" 2>&1
      status=$?
      set -e
      if [ $status -eq 124 ]; then
        echo "${label}" >> ../timed_out_runs_vt.txt
        echo "  !! TIMED OUT, logged, continuing"
      fi
    done
  done
done

python3 ../analyze_vt_sweep.py --glob "dump.atoms.*" --out ../vt_sweep_results.csv
