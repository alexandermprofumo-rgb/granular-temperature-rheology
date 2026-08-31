#!/bin/bash
# n(mu_g, k_t) on a proper 2D grid.
#
# WHY: the item-10 sensitivity spot-check found n depends on tangential
# stiffness at finite friction (mu_g=1.0: 0.190/0.281/0.208 for k_t =
# NULL/1e4/1e5, ~4 sigma) with a clean null at mu_g=0. No proposed framework
# contains tangential elasticity -- generalised isostaticity predicts
# sensitivity to mu_g but NOT k_t. It is the only new, unexplained, positive
# result in the campaign, and it was measured at 3 points.
#
# GRID PLACEMENT: for `tangential mindlin NULL`, LAMMPS sets k_t = 8*G_eff.
# With hertz/material E=1e5, nu=0.3 that is 9.05e4, and the normal prefactor
# 4/3*E_eff is 7.33e4. So the old "1e5" was 1.1x the default (hence ~= NULL)
# and "1e4" was 0.11x. The effect therefore lives BELOW the default, and the
# grid is log-spaced from 0.011x to 3.3x to bracket it.
#
# The mu_g=0 row is the control: no tangential force exists at zero friction,
# so all six k_t MUST return identical n. If they do not, the pipeline leaks.
#
# Reuses sweep_sens/ and its label convention, so the 108 runs already done at
# k_t = 1e4/1e5/NULL and mu_g = 0/0.3/1.0 are skipped rather than repeated.
#
# Usage:  ./run_ktgrid.sh [njobs]
set -u
cd "$(dirname "$0")"

NJOBS=${1:-12}
D=sweep_sens
TG=(0.0005 0.002 0.006 0.015)
MUG=(0.0 0.03 0.1 0.3 1.0 5.0)
KT=(1.0e3 3.0e3 1.0e4 3.0e4 1.0e5 3.0e5)
SEEDS=(1 2 3)
PHI=0.80; PCONF=2.0; GDOT=1.0e-3; N=4000

if command -v gtimeout >/dev/null 2>&1; then TO="gtimeout 3600"; else TO="timeout 3600"; fi
mkdir -p $D

# data files first: generated serially so parallel workers never race on them
for s in "${SEEDS[@]}"; do
  [ -f "$D/data.sens_seed${s}" ] || python3 gen_data.py --N $N --phi $PHI --seed $s --out "$D/data.sens_seed${s}"
done

one(){ # $1=mu_g $2=kt $3=T $4=seed
  local mg=$1 kt=$2 T=$3 s=$4
  local label="e0.5_k${kt}_mu${mg}_T${T}_s${s}"
  if [ -f "$D/log.${label}" ] && grep -q "^DONE" "$D/log.${label}"; then return; fi
  ( cd "$D" && $TO lmp_serial -in ../in.granular_2d_sens \
      -var datafile "data.sens_seed${s}" -var mu_g $mg -var gdot $GDOT \
      -var Pconf $PCONF -var Tgran $T -var seed $s -var e_rest 0.5 \
      -var kt_spec $kt -var nequil 40000 -var nmeas 80000 \
      -var label "$label" > "run.${label}.out" 2>&1 )
}
export -f one 2>/dev/null || true

total=0; queued=0
for mg in "${MUG[@]}"; do for kt in "${KT[@]}"; do for T in "${TG[@]}"; do for s in "${SEEDS[@]}"; do
  total=$((total+1))
  label="e0.5_k${kt}_mu${mg}_T${T}_s${s}"
  if [ -f "$D/log.${label}" ] && grep -q "^DONE" "$D/log.${label}"; then continue; fi
  queued=$((queued+1))
  one $mg $kt $T $s &
  while [ "$(jobs -rp | wc -l)" -ge "$NJOBS" ]; do wait -n 2>/dev/null || sleep 1; done
done; done; done; done
wait

echo "grid: ${total} cells, ${queued} newly run, $((total-queued)) reused"
echo "KTGRID_COMPLETE"
