#!/bin/bash
# k_t at Pconf=10, to make the stiffness decomposition clean.
#
# n depends on BOTH stiffnesses, with opposite signs at mu_g=0.3:
#   scaling Emod moves k_n and k_t together (mindlin NULL keeps the ratio
#   pinned), giving  dn/dln E = dn/dln k_n + dn/dln k_t = +0.0165;
#   the k_t grid moves k_t alone, giving  dn/dln k_t = -0.0083,
#   hence dn/dln k_n = +0.025.
#
# But those two sweeps ran at DIFFERENT pressures -- the k_t grid at Pconf=2
# (hypostatic, Z=3.30) and the Emod sweep at Pconf=10 (isostatic, Z=4.04) --
# and pressure has already been shown to change conclusions qualitatively
# (it produced the spurious "framework swap" and inflated the force-weighted
# fabric result). So the decomposition above mixes two states and cannot be
# quoted.
#
# This repeats the k_t scan at Pconf=10 so both partial derivatives are
# measured in the same state, at the same mu_g values as run_stiffness.sh.
# The result is two independent numbers per friction -- dn/dln k_n and
# dn/dln k_t -- which is the sharpest surviving constraint on any
# contact-mechanical theory, and the thing to hand a theorist.
#
# k_t values bracket the mindlin default 9.05e4 (see analyze_ktgrid.py). The
# two lowest rungs of the old grid are omitted: k_t/k_n < 0.04 is unphysical
# for elastic spheres AND numerically unstable (the tangential spring stores
# huge energy before slipping; Theta/Tgran reached 5000).
#
# Usage:  ./run_ktgrid_P10.sh [njobs]     (default 12)
set -u
cd "$(dirname "$0")"

NJOBS=${1:-12}
D=sweep_kt_P10
KT=(1.0e4 3.0e4 1.0e5 3.0e5)
MUG=(0.1 0.3 1.0)
TG=(0.001 0.003 0.008 0.02)
SEEDS=(1 2 3)
N=4000; PHI=0.80; PCONF=10.0; GDOT=1.0e-3

if command -v gtimeout >/dev/null 2>&1; then TO="gtimeout 5400"; else TO="timeout 5400"; fi
mkdir -p $D
for s in "${SEEDS[@]}"; do
  [ -f "$D/data.ktP10_seed${s}" ] || python3 gen_data.py --N $N --phi $PHI --seed $s --out "$D/data.ktP10_seed${s}"
done

one(){
  local kt=$1 mg=$2 T=$3 s=$4
  local label="k${kt}_mu${mg}_T${T}_s${s}"
  if [ -f "$D/log.${label}" ] && grep -q "^DONE" "$D/log.${label}"; then return; fi
  ( cd "$D" && $TO lmp_serial -in ../in.granular_2d_sens \
      -var datafile "data.ktP10_seed${s}" -var mu_g $mg -var gdot $GDOT \
      -var Pconf $PCONF -var Tgran $T -var seed $s -var e_rest 0.5 \
      -var kt_spec $kt -var nequil 40000 -var nmeas 80000 \
      -var label "$label" > "run.${label}.out" 2>&1 )
}

for kt in "${KT[@]}"; do for mg in "${MUG[@]}"; do for T in "${TG[@]}"; do for s in "${SEEDS[@]}"; do
  one $kt $mg $T $s &
  while [ "$(jobs -rp | wc -l)" -ge "$NJOBS" ]; do wait -n 2>/dev/null || sleep 1; done
done; done; done; done
wait
echo "KTGRID_P10_COMPLETE"
