#!/bin/bash
# 3D force-resolved contact-tracking sweep.
#
# Target: the joint 2D+3D Delta-Z_eff collapse test. The mu_g and Theta grids
# MATCH the published 3D macroscopic sweep (run_sweep_3d.sh) so that n(mu_g)
# and Delta-Z_eff(mu_g) exist at identical parameter values, which is what the
# collapse requires.
#
# Packing uses the calibrated 3D state phi=0.64, Pconf=10.0 (validated here:
# Z = 5.89 against the 3D frictionless isostatic value of 6).
#
# Seed is $1 so seeds run concurrently; labels are seed-distinct.
#
# Usage: ./run_tracking_forces_3d.sh 1
set -e

SEED=$1
if [ -z "$SEED" ]; then echo "usage: $0 <seed>"; exit 1; fi

if command -v timeout >/dev/null 2>&1; then TIMEOUT_CMD="timeout 1800"
elif command -v gtimeout >/dev/null 2>&1; then TIMEOUT_CMD="gtimeout 1800"
else TIMEOUT_CMD=""; fi

MU_G_VALUES=(0.0 0.05 0.1 0.15 0.2 0.3 0.5 1.0)
TGRAN_VALUES=(0.0005 0.001 0.003 0.008 0.02)

N=2000
PHI=0.64
PCONF=10.0
GDOT=1.0e-3
NEQUIL=40000
NTRACK=3000
CDUMP_EVERY=50

mkdir -p sweep_tracking_forces_3d
cd sweep_tracking_forces_3d

datafile="data.trk3d_seed${SEED}"
if [ ! -f "$datafile" ]; then
  python3 ../gen_data_3d.py --N ${N} --phi ${PHI} --seed ${SEED} --out "$datafile"
fi

for mu_g in "${MU_G_VALUES[@]}"; do
  for Tgran in "${TGRAN_VALUES[@]}"; do
    label="mu${mu_g}_T${Tgran}_s${SEED}"
    if [ -f "log.${label}" ] && grep -q "^DONE" "log.${label}"; then
      echo "skip (done): ${label}"; continue
    fi
    echo "=== running ${label} ==="
    set +e
    ${TIMEOUT_CMD} lmp_serial -in ../in.contact_tracking_forces_3d \
        -var datafile "$datafile" -var mu_g ${mu_g} -var gdot ${GDOT} \
        -var Pconf ${PCONF} -var Tgran ${Tgran} -var seed ${SEED} \
        -var label "$label" -var nequil ${NEQUIL} -var ntrack ${NTRACK} \
        -var cdump_every ${CDUMP_EVERY} \
        > "run.${label}.out" 2>&1
    status=$?
    set -e
    if [ $status -eq 124 ]; then
      echo "${label}" >> ../timed_out_tracking3d.txt
    fi
  done
done
echo "SEED ${SEED} COMPLETE"
