#!/bin/bash
# Force-resolved tracking at varying k_t -- the decisive test of R_E.
#
# R_E = (f_t^2/k_t)/(f_n^2/k_n), the ratio of stored tangential to normal
# elastic energy, is currently the best candidate for organising n: on 13
# matched 2D rows it gives chi2/dof = 0.48 against 1.12 for Z, 1.29 for chi
# and 4.9-6.1 for the fabric and force-distribution measures (dBIC ~ 5.7 over
# the next best). It is also the only measured quantity in which k_n and k_t
# appear with OPPOSITE powers, which is what the opposite-signed stiffness
# derivatives demand -- the motivation preceded the fit.
#
# But its two-lever test is nearly powerless as things stand. R_E spans 80x
# along the friction lever and only 13% along the Emod lever, so "survives at
# 1.4 sigma" mostly reflects the absence of orthogonal variation. chi passed
# an equally weak local test and then failed globally; the lesson is to test
# a candidate with a lever that moves it HARD.
#
# k_t is that lever: it enters R_E directly, at fixed mu_g and fixed k_n.
# The 144 existing k_t runs cannot be used -- in.granular_2d_sens dumps only
# `dx dy fx fy`, with no tangential force, so R_E is not recoverable from
# them. Hence this sweep, which uses the force-resolved input (now
# parameterised by kt_spec) over the same k_t values as sweep_kt_P10 so the
# macroscopic n values already measured there can be reused directly.
#
# Usage:  ./run_tracking_kt.sh [njobs]
set -u
cd "$(dirname "$0")"

NJOBS=${1:-6}
D=sweep_tracking_kt
KT=(1.0e4 3.0e4 1.0e5 3.0e5)
MUG=(0.1 0.3 1.0)
TG=(0.001 0.003 0.008)
SEEDS=(1 2 3)
N=4000; PHI=0.80; PCONF=10.0; GDOT=1.0e-3

if command -v gtimeout >/dev/null 2>&1; then TO="gtimeout 5400"; else TO="timeout 5400"; fi
mkdir -p $D
for s in "${SEEDS[@]}"; do
  [ -f "$D/data.tkt_seed${s}" ] || python3 gen_data.py --N $N --phi $PHI --seed $s --out "$D/data.tkt_seed${s}"
done

one(){
  local kt=$1 mg=$2 T=$3 s=$4
  local label="k${kt}_mu${mg}_T${T}_s${s}"
  if [ -f "$D/log.${label}" ] && grep -q "^DONE" "$D/log.${label}"; then return; fi
  ( cd "$D" && $TO lmp_serial -in ../in.contact_tracking_forces_2d \
      -var datafile "data.tkt_seed${s}" -var mu_g $mg -var gdot $GDOT \
      -var Pconf $PCONF -var Tgran $T -var seed $s -var label "$label" \
      -var kt_spec $kt -var nequil 40000 -var ntrack 3000 \
      -var dump_every 15 -var cdump_every 30 > "run.${label}.out" 2>&1 )
}

for kt in "${KT[@]}"; do for mg in "${MUG[@]}"; do for T in "${TG[@]}"; do for s in "${SEEDS[@]}"; do
  one $kt $mg $T $s &
  while [ "$(jobs -rp | wc -l)" -ge "$NJOBS" ]; do wait -n 2>/dev/null || sleep 1; done
done; done; done; done
wait
echo "TRACKING_KT_COMPLETE"
