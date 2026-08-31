#!/bin/bash
# THE INERTIAL-NUMBER SCAN -- the project's largest untested assumption.
#
# Every sweep in this project ran at gdot = 1.0e-3. The measured spread in I is
# 1.1x and is incidental pressure drift, not a scan. So in the law this paper
# is about,
#                        mu * Theta^n = F(I),
# F(I) has never been tested and it is not known whether n depends on I at all.
# Every number in the constraint list is a single-I measurement. A referee will
# ask this first, and at present the honest answer is that we cannot say
# whether n is a material property or a value at one operating point.
#
# DESIGN. gdot is the only knob that moves I = gdot d sqrt(rho/P) at fixed
# pressure, so it is scanned over a factor of 10 in I:
#     gdot = 3.162e-4, (1.0e-3 already run), 3.162e-3
# at four frictions spanning the peak, over the production Theta grid.
#
# STRAIN IS HELD FIXED, NOT STEP COUNT. Strain per step is dt*gdot, so a run at
# one third the shear rate covers one third the strain in the same number of
# steps and may never reach steady state. Step counts are therefore scaled as
# 1/gdot, which is the whole reason the slow arm of this scan is expensive:
#     gdot = 3.162e-4 -> nequil 126000, nmeas 253000   (3.16x)
#     gdot = 3.162e-3 -> nequil  12650, nmeas  25300   (0.32x)
#
# THETA GRID EXTENDED WARM. Shear heating sets a floor on the accessible
# Theta, and that floor RISES with gdot, so the fast arm cannot reach the
# coldest setpoints. Two warm setpoints (0.06, 0.12) are added so that a common
# Theta window survives across all three shear rates -- without overlap the
# comparison is an extrapolation. The gates decide what is usable.
#
# Uses in.granular_2d, which also writes contact dumps, so the same runs test
# whether the CHANNEL decomposition is I-dependent -- not just n.
set -u
cd "$(dirname "$0")"
NJOBS=${1:-14}
mkdir -p sweep_iscan
for s in 1 2; do
  [ -f "sweep_iscan/data.m2d_seed${s}" ] || cp "sweep_matched2d/data.m2d_seed${s}" "sweep_iscan/data.m2d_seed${s}"
done

launch(){ local lab=$1; shift
  [ -f "sweep_iscan/log.${lab}" ] && grep -q "^DONE" "sweep_iscan/log.${lab}" && return
  ( cd sweep_iscan && lmp_serial "$@" -var label "$lab" > "run.${lab}.out" 2>&1 ) &
  while [ "$(jobs -rp | wc -l)" -ge "$NJOBS" ]; do wait -n 2>/dev/null || sleep 1; done
}

for G in 3.162e-4 3.162e-3; do
  case $G in
    3.162e-4) NEQ=126000; NME=253000 ;;
    3.162e-3) NEQ=12650;  NME=25300  ;;
  esac
  for mg in 0.0 0.15 0.3 1.0; do
   for T in 0.0005 0.001 0.0015 0.003 0.005 0.008 0.013 0.02 0.03 0.06 0.12; do
    for s in 1 2; do
      launch "g${G}_mu${mg}_T${T}_s${s}" -in ../in.granular_2d \
        -var datafile "data.m2d_seed${s}" -var mu_g $mg -var gdot $G \
        -var Pconf 10.0 -var Tgran $T -var seed $s \
        -var nequil $NEQ -var nmeas $NME
done; done; done; done

wait
echo "ISCAN_COMPLETE"
