#!/bin/bash
# Extend the 3D sweeps to intermediate/high friction, to POWER the dZ_eff
# collapse test rather than merely pass it.
#
# With 2D now run at the 3D pressure, the two dimensions finally overlap in
# dZ_eff (window width 0.90, up from 0.27), and n collapses onto dZ_eff at
# 1.7 sigma -- much better than mu_g (2.9) or chi (3.5). But the overlap
# window [-0.025, +0.870] contains only TWO explicit 3D points (mu_g=0 at
# dZ=0.133 and mu_g=0.5 at dZ=0.358); everything between dZ=0.358 and 0.882
# is interpolation across a gap. So the test is under-powered and the 1.7
# sigma should not be quoted as a pass.
#
# The existing 3D grid jumps mu_g 0.3 -> 0.5 -> 1.0, which is exactly where
# dZ_eff falls fastest (0.882 -> 0.358 -> -0.086). Adding mu_g in that gap,
# plus two higher values, fills the overlap window where the collapse is
# actually being tested.
#
# Runs BOTH the macroscopic sweep (for n) and the force-resolved tracking
# (for chi and dZ_eff); the collapse needs the pair at the same mu_g.
#
# Usage:  ./run_3d_extend.sh [njobs]     (default 12)
set -u
cd "$(dirname "$0")"

NJOBS=${1:-12}
MUG=(0.4 0.6 0.8 1.5 2.0)
TG_MACRO=(0.0005 0.001 0.003 0.008 0.02)
TG_TRACK=(0.001 0.003 0.008)
SEEDS=(1 2 3)
N=4000; PHI=0.64; PCONF=10.0; GDOT=1.0e-3

if command -v gtimeout >/dev/null 2>&1; then TO="gtimeout 3600"; else TO="timeout 3600"; fi
mkdir -p sweep_run_3d sweep_tracking_forces_3d
for s in "${SEEDS[@]}"; do
  [ -f "sweep_run_3d/data.granular3d_seed${s}" ] || \
    python3 gen_data_3d.py --N $N --phi $PHI --seed $s --out "sweep_run_3d/data.granular3d_seed${s}"
  [ -f "sweep_tracking_forces_3d/data.trk3d_seed${s}" ] || \
    python3 gen_data_3d.py --N $N --phi $PHI --seed $s --out "sweep_tracking_forces_3d/data.trk3d_seed${s}"
done

macro(){
  local mg=$1 T=$2 s=$3 label="mu${mg}_T${T}_s${s}"
  [ -f "sweep_run_3d/log.${label}" ] && grep -q "^DONE" "sweep_run_3d/log.${label}" && return
  ( cd sweep_run_3d && $TO lmp_serial -in ../in.granular_3d \
      -var datafile "data.granular3d_seed${s}" -var mu_g $mg -var gdot $GDOT \
      -var Pconf $PCONF -var Tgran $T -var seed $s -var label "$label" \
      -var nequil 40000 -var nmeas 80000 > "run.${label}.out" 2>&1 )
}
track(){
  local mg=$1 T=$2 s=$3 label="mu${mg}_T${T}_s${s}"
  [ -f "sweep_tracking_forces_3d/log.${label}" ] && grep -q "^DONE" "sweep_tracking_forces_3d/log.${label}" && return
  ( cd sweep_tracking_forces_3d && $TO lmp_serial -in ../in.contact_tracking_forces_3d \
      -var datafile "data.trk3d_seed${s}" -var mu_g $mg -var gdot $GDOT \
      -var Pconf $PCONF -var Tgran $T -var seed $s -var label "$label" \
      -var nequil 40000 -var ntrack 3000 -var dump_every 15 -var cdump_every 30 \
      > "run.${label}.out" 2>&1 )
}

for mg in "${MUG[@]}"; do
  for T in "${TG_MACRO[@]}"; do for s in "${SEEDS[@]}"; do
    macro $mg $T $s &
    while [ "$(jobs -rp | wc -l)" -ge "$NJOBS" ]; do wait -n 2>/dev/null || sleep 1; done
  done; done
  for T in "${TG_TRACK[@]}"; do for s in "${SEEDS[@]}"; do
    track $mg $T $s &
    while [ "$(jobs -rp | wc -l)" -ge "$NJOBS" ]; do wait -n 2>/dev/null || sleep 1; done
  done; done
done
wait
echo "EXTEND3D_COMPLETE"
