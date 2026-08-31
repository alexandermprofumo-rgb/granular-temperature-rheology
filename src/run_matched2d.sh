#!/bin/bash
# 2D macroscopic sweep at the 3D operating point: Pconf=10, gdot=1.0e-3.
#
# WHY: every cross-dimension statement in the campaign compares a 2D sweep at
# Pconf=2 against a 3D sweep at Pconf=10. Because both barostat the y box
# length, Pconf -- not the phi passed to gen_data.py -- sets steady-state
# density and hence coordination. Calibration (in.calib_Z_2d) gives, at mu_g=0:
#
#     Pconf   2      5      10     20
#     Z       3.297  3.800  4.036  4.215
#
# So 2D at Pconf=2 is HYPOstatic (Z < 4) while 3D at Pconf=10 is at/above its
# own isostatic value of 6. At Pconf=10 BOTH dimensions sit at their isostatic
# point. The reported "framework swap" -- generalised isostaticity works in 3D,
# force-weighted fabric works in 2D -- may therefore be a pressure artifact
# rather than a dimensional fact. This sweep removes the confound.
#
# NOT the failed pressure lever. That failed because holding I fixed at the 2D
# value forces gdot ~ sqrt(P), and the extra shear heating overwhelms the
# thermostat. Here gdot is UNCHANGED at 1.0e-3 -- the same value the 3D sweep
# used at this same Pconf -- so the operating point is one already shown to
# work. I lands at the 3D value rather than the 2D one, which is the point.
#
# Grid matches run_sweep_3d.sh exactly (same mu_g, same Tgran, same seeds) so
# the two dimensions can be fitted and compared without interpolation.
#
# Usage:  ./run_matched2d.sh [njobs]     (default 5; the k_t grid holds 12)
set -u
cd "$(dirname "$0")"

NJOBS=${1:-5}
D=sweep_matched2d
TG=(0.0005 0.001 0.003 0.008 0.02)
MUG=(0.0 0.05 0.1 0.15 0.2 0.3 0.5 1.0)
SEEDS=(1 2 3)
PHI=0.80; PCONF=10.0; GDOT=1.0e-3; N=4000

if command -v gtimeout >/dev/null 2>&1; then TO="gtimeout 3600"; else TO="timeout 3600"; fi
mkdir -p $D
for s in "${SEEDS[@]}"; do
  [ -f "$D/data.m2d_seed${s}" ] || python3 gen_data.py --N $N --phi $PHI --seed $s --out "$D/data.m2d_seed${s}"
done

one(){
  local mg=$1 T=$2 s=$3
  local label="mu${mg}_T${T}_s${s}"
  if [ -f "$D/log.${label}" ] && grep -q "^DONE" "$D/log.${label}"; then return; fi
  ( cd "$D" && $TO lmp_serial -in ../in.granular_2d \
      -var datafile "data.m2d_seed${s}" -var mu_g $mg -var gdot $GDOT \
      -var Pconf $PCONF -var Tgran $T -var seed $s \
      -var nequil 40000 -var nmeas 80000 \
      -var label "$label" > "run.${label}.out" 2>&1 )
}

for mg in "${MUG[@]}"; do for T in "${TG[@]}"; do for s in "${SEEDS[@]}"; do
  one $mg $T $s &
  while [ "$(jobs -rp | wc -l)" -ge "$NJOBS" ]; do wait -n 2>/dev/null || sleep 1; done
done; done; done
wait
echo "MATCHED2D_COMPLETE"
