#!/bin/bash
# Curve-against-curve at MATCHED kappa, reached two different ways.
#
# The peak-position proxy r = n(0.3)/n(0.1) collapses on kappa across both
# levers (pressure vs stiffness agree to 0.04 at kappa = 3e3 and 1e4), so the
# SHAPE of n(mu_g) is kappa-organised. But at the highest kappa the
# AMPLITUDES disagree: at kappa ~ 2-3e4 the shape ratios match (1.03 vs 1.01)
# while n(0.3) is 0.223 (pressure, P=5) against 0.165 (stiffness, E=3e5).
#
# That kappa is reached on the pressure lever only at P=5 -- the closest
# usable point to the jamming boundary (Z = 3.26 there vs 3.79 at P=50; P=2
# is outright inaccessible). So the 4.9 sigma kappa "failure" reported at
# mu_g=0.3 may be jamming proximity at the low-pressure end rather than
# anything friction-specific.
#
# This run settles it: the SAME dense mu_g grid used at P=5, but reached by
# stiffening grains at P=10 (Emod = 2e5 -> kappa = 2e4) instead of lowering
# pressure. Two curves at identical kappa; if they coincide, kappa organises
# n and the earlier claim dissolves.
#
# dt and step counts scale as E^(-2/5) (in.granular_2d_stiff.py) so the run
# covers the same strain as every other sweep.
set -u
cd "$(dirname "$0")"
NJOBS=${1:-14}
D=sweep_kappa_match
EMOD=2.0e5
MUG=(0.03 0.05 0.07 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.5 1.0)
TG=(0.001 0.003 0.008 0.02)
SEEDS=(1 2 3)
N=4000; PHI=0.80; PCONF=10.0; GDOT=1.0e-3
if command -v gtimeout >/dev/null 2>&1; then TO="gtimeout 7200"; else TO="timeout 7200"; fi
mkdir -p $D
for s in "${SEEDS[@]}"; do
  [ -f "$D/data.km_seed${s}" ] || python3 gen_data.py --N $N --phi $PHI --seed $s --out "$D/data.km_seed${s}"
done
read dt nr ne nm < <(python3 in.granular_2d_stiff.py $EMOD)
for mg in "${MUG[@]}"; do for T in "${TG[@]}"; do for s in "${SEEDS[@]}"; do
  label="mu${mg}_T${T}_s${s}"
  if [ -f "$D/log.${label}" ] && grep -q "^DONE" "$D/log.${label}"; then continue; fi
  ( cd "$D" && $TO lmp_serial -in ../in.granular_2d_stiff \
      -var datafile "data.km_seed${s}" -var mu_g $mg -var gdot $GDOT \
      -var Pconf $PCONF -var Tgran $T -var seed $s -var Emod $EMOD \
      -var dt $dt -var nrelax $nr -var nequil $ne -var nmeas $nm \
      -var label "$label" > "run.${label}.out" 2>&1 ) &
  while [ "$(jobs -rp | wc -l)" -ge "$NJOBS" ]; do wait -n 2>/dev/null || sleep 1; done
done; done; done
wait
echo "KAPPA_MATCH_COMPLETE"
