#!/bin/bash
# Force-resolved contact tracking in 2D at the 3D operating point (Pconf=10).
#
# WHY: chi (the Coulomb-mobilised fraction), Z, dZ_eff and the force-weighted
# fabric a_c^w have only ever been measured in 2D at Pconf=2 -- where the
# packing is hypostatic (Z=3.30 < 4) and, as the macroscopic refit showed,
# the fitted exponent is unstable to the analysis window. The 3D tracking
# sweep is already at Pconf=10. So every 2D-vs-3D microstructural claim in the
# campaign (the "framework swap", the dZ_eff collapse) compares unlike states.
#
# This regenerates the 2D microstructure at Pconf=10 so that both dimensions
# are isostatic and directly comparable, and so the chi-collapse hypothesis
# -- that n is controlled by the sticking fraction rather than by mu_g or
# dimension -- becomes testable. Right now 2D peaks at chi=0.017 and 3D at
# chi=0.122, but the 2D number comes from the fragile Pconf=2 sweep.
#
# Grid matches run_sweep_3d.sh in mu_g. N=4000 (not the 2000 used by the
# original tracking sweep) so a_c is directly comparable to the production
# macroscopic runs -- the old N mismatch is the known ~25% a_c discrepancy.
# The coldest setpoint is omitted: it violates fixed-I in every sweep.
#
# Usage:  ./run_tracking_matched.sh [njobs]     (default 12)
set -u
cd "$(dirname "$0")"

NJOBS=${1:-12}
D=sweep_tracking_matched2d
MUG=(0.0 0.05 0.1 0.15 0.2 0.3 0.5 1.0)
TG=(0.001 0.003 0.008)
SEEDS=(1 2 3)
N=4000; PHI=0.80; PCONF=10.0; GDOT=1.0e-3
NEQUIL=40000; NTRACK=3000; DUMP_EVERY=15; CDUMP_EVERY=30

if command -v gtimeout >/dev/null 2>&1; then TO="gtimeout 1800"; else TO="timeout 1800"; fi
mkdir -p $D
for s in "${SEEDS[@]}"; do
  [ -f "$D/data.trkm_seed${s}" ] || python3 gen_data.py --N $N --phi $PHI --seed $s --out "$D/data.trkm_seed${s}"
done

one(){
  local mg=$1 T=$2 s=$3
  local label="mu${mg}_T${T}_s${s}"
  if [ -f "$D/log.${label}" ] && grep -q "^DONE" "$D/log.${label}"; then return; fi
  ( cd "$D" && $TO lmp_serial -in ../in.contact_tracking_forces_2d \
      -var datafile "data.trkm_seed${s}" -var mu_g $mg -var gdot $GDOT \
      -var Pconf $PCONF -var Tgran $T -var seed $s -var label "$label" \
      -var nequil $NEQUIL -var ntrack $NTRACK \
      -var dump_every $DUMP_EVERY -var cdump_every $CDUMP_EVERY \
      > "run.${label}.out" 2>&1 )
}

for mg in "${MUG[@]}"; do for T in "${TG[@]}"; do for s in "${SEEDS[@]}"; do
  one $mg $T $s &
  while [ "$(jobs -rp | wc -l)" -ge "$NJOBS" ]; do wait -n 2>/dev/null || sleep 1; done
done; done; done
wait
echo "TRACKING_MATCHED_COMPLETE"
