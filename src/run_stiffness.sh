#!/bin/bash
# Phase 2: is the k_t dependence a stiffness RATIO effect or a contact
# DURATION effect?  Scan the material modulus at FIXED k_t/k_n.
#
# n depends on tangential stiffness (432-cell grid) and on the thermostat
# damping time (Tier 0-A). Both are timescales. Using `tangential mindlin
# NULL` makes LAMMPS derive k_t = 8*G_eff from Emod, so scaling Emod moves
# k_n and k_t together -- the ratio stays pinned at 1.235 -- while the
# Hertzian contact time moves as E^(-2/5). So:
#
#   n varies here  ->  contact DURATION matters; the k_t and tdamp
#                      dependences are plausibly one effect.
#   n flat here    ->  the k_t dependence is genuinely tangential ELASTICITY,
#                      and tdamp is a separate problem.
#
# SECOND PURPOSE. The smoke test showed Z falls from 4.04 (E=1e5) to 3.17
# (E=1e6) at fixed Pconf and mu_g -- stiffer grains, smaller overlaps, fewer
# contacts. That is an INDEPENDENT lever on Z, and hence on dZ_eff, that does
# not touch mu_g. The dZ_eff collapse currently sits at 1.5 sigma on 8 points
# where dZ_eff is only movable via friction; this sweep can move it
# orthogonally, which is a far more stringent test of the collapse than
# anything run so far. (Contrast the pressure lever, which failed because
# holding I fixed forced gdot ~ sqrt(P) and shear heating won. Emod does not
# appear in I at all, so there is no such conflict.)
#
# Timestep and step counts are scaled per modulus by in.granular_2d_stiff.py
# so every run covers the same physical duration and the same strain.
#
# Usage:  ./run_stiffness.sh [njobs]     (default 12)
set -u
cd "$(dirname "$0")"

NJOBS=${1:-12}
D=sweep_stiffness
EMOD=(1.0e4 3.0e4 1.0e5 3.0e5 1.0e6)
MUG=(0.1 0.3 1.0)
TG=(0.001 0.003 0.008 0.02)
SEEDS=(1 2 3)
N=4000; PHI=0.80; PCONF=10.0; GDOT=1.0e-3

if command -v gtimeout >/dev/null 2>&1; then TO="gtimeout 5400"; else TO="timeout 5400"; fi
mkdir -p $D
for s in "${SEEDS[@]}"; do
  [ -f "$D/data.stiff_seed${s}" ] || python3 gen_data.py --N $N --phi $PHI --seed $s --out "$D/data.stiff_seed${s}"
done

one(){
  local E=$1 mg=$2 T=$3 s=$4
  local label="E${E}_mu${mg}_T${T}_s${s}"
  if [ -f "$D/log.${label}" ] && grep -q "^DONE" "$D/log.${label}"; then return; fi
  read dt nr ne nm < <(python3 in.granular_2d_stiff.py "$E")
  ( cd "$D" && $TO lmp_serial -in ../in.granular_2d_stiff \
      -var datafile "data.stiff_seed${s}" -var mu_g $mg -var gdot $GDOT \
      -var Pconf $PCONF -var Tgran $T -var seed $s -var Emod $E \
      -var dt $dt -var nrelax $nr -var nequil $ne -var nmeas $nm \
      -var label "$label" > "run.${label}.out" 2>&1 )
}

for E in "${EMOD[@]}"; do for mg in "${MUG[@]}"; do for T in "${TG[@]}"; do for s in "${SEEDS[@]}"; do
  one $E $mg $T $s &
  while [ "$(jobs -rp | wc -l)" -ge "$NJOBS" ]; do wait -n 2>/dev/null || sleep 1; done
done; done; done; done
wait
echo "STIFFNESS_COMPLETE"
