#!/bin/bash
# Force-resolved contact-tracking sweep (Tier 1 enabler).
#
# Regenerates the contact-tracking data that was not preserved from the
# original campaign, and additionally captures per-contact normal and
# tangential forces so that the Tier 1 analyses become possible:
#   - strong/weak network split of the reorientation rate
#   - Coulomb-mobilised fraction chi, hence Z_c(chi) and dZ_eff = Z - Z_c
#   - force-weighted fabric tensor F^w ~ sum f_n n n
#   - force distribution statistics P(f)
#
# The mu_g grid deliberately MATCHES the macroscopic 2D sweep (results.csv)
# so that n(mu_g) and dZ_eff(mu_g) are available at identical friction
# values, which is what the dZ_eff collapse test needs. Theta grid matches
# the core fitting window.
#
# Seed is taken as $1 so seeds can run concurrently (labels are
# seed-distinct). Run analysis separately once all seeds finish.
#
# Usage: ./run_tracking_forces_sweep.sh 1
set -e

SEED=$1
if [ -z "$SEED" ]; then echo "usage: $0 <seed>"; exit 1; fi

if command -v timeout >/dev/null 2>&1; then TIMEOUT_CMD="timeout 1800"
elif command -v gtimeout >/dev/null 2>&1; then TIMEOUT_CMD="gtimeout 1800"
else TIMEOUT_CMD=""; fi

MU_G_VALUES=(0.0 0.001 0.01 0.03 0.1 0.3 1.0 2.0 5.0)
TGRAN_VALUES=(0.0005 0.002 0.006 0.015)

N=2000
PHI=0.80
PCONF=2.0
GDOT=1.0e-3
NEQUIL=40000
NTRACK=3000
DUMP_EVERY=15
CDUMP_EVERY=30

mkdir -p sweep_tracking_forces
cd sweep_tracking_forces

datafile="data.trk_seed${SEED}"
if [ ! -f "$datafile" ]; then
  python3 ../gen_data.py --N ${N} --phi ${PHI} --seed ${SEED} --out "$datafile"
fi

for mu_g in "${MU_G_VALUES[@]}"; do
  for Tgran in "${TGRAN_VALUES[@]}"; do
    label="mu${mu_g}_T${Tgran}_s${SEED}"
    if [ -f "log.${label}" ] && grep -q "^DONE" "log.${label}"; then
      echo "skip (done): ${label}"; continue
    fi
    echo "=== running ${label} ==="
    set +e
    ${TIMEOUT_CMD} lmp_serial -in ../in.contact_tracking_forces_2d \
        -var datafile "$datafile" -var mu_g ${mu_g} -var gdot ${GDOT} \
        -var Pconf ${PCONF} -var Tgran ${Tgran} -var seed ${SEED} \
        -var label "$label" -var nequil ${NEQUIL} -var ntrack ${NTRACK} \
        -var dump_every ${DUMP_EVERY} -var cdump_every ${CDUMP_EVERY} \
        > "run.${label}.out" 2>&1
    status=$?
    set -e
    if [ $status -eq 124 ]; then
      echo "${label}" >> ../timed_out_tracking.txt
      echo "  !! TIMED OUT, logged, continuing"
    fi
  done
done
echo "SEED ${SEED} COMPLETE"
