#!/bin/bash
# Full sweep driver: for each seed, generate one initial packing, then loop
# over mu_g and Tgran (independent friction / temperature scan at fixed
# shear rate gdot, i.e. fixed inertial number I), running LAMMPS for each
# combination. Finishes by calling analyze_sweep.py to fit n(mu_g).
#
# Edit the arrays below to trade off resolution vs runtime. Defaults here
# are a genuine (not toy) sweep, but will take a while -- see the
# "quick" variant invocation instructions at the bottom of this file for
# a fast sanity-check version.

set -e

# macOS doesn't ship GNU 'timeout' by default; use 'gtimeout' from
# `brew install coreutils` if 'timeout' isn't found, else run without one.
if command -v timeout >/dev/null 2>&1; then
  TIMEOUT_CMD="timeout 1800"
elif command -v gtimeout >/dev/null 2>&1; then
  TIMEOUT_CMD="gtimeout 1800"
else
  echo "WARNING: no 'timeout'/'gtimeout' found -- runs won't be time-capped."
  echo "  (brew install coreutils gives you gtimeout)"
  TIMEOUT_CMD=""
fi

MU_G_VALUES=(0.0 0.001 0.01 0.03 0.1 0.3 1.0 2.0 5.0)
TGRAN_VALUES=(0.0002 0.0005 0.002 0.006 0.015 0.03 0.045)
SEEDS=(1 2 3)

N=4000
PHI=0.80
GDOT=1.0e-3
PCONF=2.0
NEQUIL=40000
NMEAS=80000

mkdir -p sweep_run
cd sweep_run

for seed in "${SEEDS[@]}"; do
  datafile="data.granular_seed${seed}"
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
      ${TIMEOUT_CMD} lmp_serial -in ../in.granular_2d \
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
        echo "${label}" >> ../timed_out_runs.txt
        echo "  !! TIMED OUT after 30 min, logged to timed_out_runs.txt, continuing"
      fi
    done
  done
done

python3 ../analyze_sweep.py --out ../results.csv

# Sanity-check variant:
#
#   mkdir -p quicktest && cd quicktest
#   python3 ../gen_data.py --N 500 --phi 0.85 --seed 1 --out data.q
#   for mu_g in 0.0 0.3 1.0; do
#     for Tgran in 0.005 0.02 0.05; do
#       lmp_serial -in ../in.granular_2d -var datafile data.q -var mu_g $mu_g \
#           -var gdot 1.0e-3 -var Tgran $Tgran -var seed 1 \
#           -var label "mu${mu_g}_T${Tgran}_s1" \
#           -var nequil 4000 -var nmeas 4000 -log none
#     done
#   done
#   python3 ../analyze_sweep.py --out ../quicktest_results.csv
