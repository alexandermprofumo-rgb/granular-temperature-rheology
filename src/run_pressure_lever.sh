#!/bin/bash
# PRESSURE LEVER sweep -- makes the dZ_eff collapse test decidable.
#
# The 2D and 3D sweeps occupy nearly disjoint ranges of dZ_eff (2D mostly
# hypostatic, 3D mostly hyperstatic), so the joint collapse cannot be tested
# as they stand. Confining pressure is an INDEPENDENT lever on dZ_eff:
# measured at mu_g=0.3, raising Pconf 2 -> 50 moves dZ_eff +0.204 -> +0.712
# (Z rises 3.33 -> 3.74 while chi falls 0.124 -> 0.023, so Z_c falls too),
# which reaches into the 3D range.
#
# This gives a SHARPER test than the joint-dimension collapse: if n is a
# function of dZ_eff alone, then n(mu_g, Pconf) must collapse onto one curve.
# If n stays put while dZ_eff moves with pressure, the collapse is falsified
# directly.
#
# CRITICAL CONTROL: I = gdot/sqrt(P), and n is only well defined at fixed I.
# So gdot is scaled as sqrt(Pconf/2) to hold I = 7.07e-4 constant across all
# pressures -- otherwise a change in n could be an I effect, not a dZ_eff one.
#
# $1 = seed, $2 = mode ("macro" for n, "track" for dZ_eff)
set -e
SEED=$1; MODE=$2
if [ -z "$SEED" ] || [ -z "$MODE" ]; then echo "usage: $0 <seed> <macro|track>"; exit 1; fi
if command -v gtimeout >/dev/null 2>&1; then TO="gtimeout 2400"; else TO="timeout 2400"; fi

PCONF_VALUES=(2.0 10.0 50.0)
MU_G_VALUES=(0.0 0.1 0.3 1.0)
TGRAN_VALUES=(0.0005 0.002 0.006 0.015)
PHI=0.80

if [ "$MODE" = "macro" ]; then N=4000; DIR=sweep_plever_macro; else N=2000; DIR=sweep_plever_track; fi
mkdir -p $DIR; cd $DIR
datafile="data.pl_${MODE}_seed${SEED}"
[ -f "$datafile" ] || python3 ../gen_data.py --N ${N} --phi ${PHI} --seed ${SEED} --out "$datafile"

for Pconf in "${PCONF_VALUES[@]}"; do
  GDOT=$(python3 -c "import math;print(f'{1.0e-3*math.sqrt($Pconf/2.0):.6e}')")
  for mu_g in "${MU_G_VALUES[@]}"; do
    for Tgran in "${TGRAN_VALUES[@]}"; do
      label="P${Pconf}_mu${mu_g}_T${Tgran}_s${SEED}"
      if [ -f "log.${label}" ] && grep -q "^DONE" "log.${label}"; then echo "skip: ${label}"; continue; fi
      echo "=== ${MODE} ${label} (gdot=${GDOT}) ==="
      set +e
      if [ "$MODE" = "macro" ]; then
        $TO lmp_serial -in ../in.granular_2d -var datafile "$datafile" -var mu_g ${mu_g} \
            -var gdot ${GDOT} -var Pconf ${Pconf} -var Tgran ${Tgran} -var seed ${SEED} \
            -var label "$label" -var nequil 40000 -var nmeas 80000 > "run.${label}.out" 2>&1
      else
        $TO lmp_serial -in ../in.contact_tracking_forces_2d -var datafile "$datafile" \
            -var mu_g ${mu_g} -var gdot ${GDOT} -var Pconf ${Pconf} -var Tgran ${Tgran} \
            -var seed ${SEED} -var label "$label" -var nequil 40000 -var ntrack 3000 \
            -var dump_every 15 -var cdump_every 50 > "run.${label}.out" 2>&1
      fi
      set -e
    done
  done
done
echo "SEED ${SEED} ${MODE} COMPLETE"
