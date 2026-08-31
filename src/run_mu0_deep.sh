#!/bin/bash
# IS n(mu_g = 0) EXACTLY ZERO?
#
# The steady campaign gives n(mu_g=0) = 0.0226 +/- 0.0168 (2D) and
# 0.0192 +/- 0.0076 (3D) -- combined 0.0198 +/- 0.0069, about 3 sigma from
# zero and ~10x below the geometric prediction of 1/(2D). Whether it is small
# or actually ZERO is the sharpest question left: if zero, "there is no fabric
# erasure without friction" replaces "the geometric theory is off by a factor
# of ten", the whole phenomenon is frictional in origin, and the search space
# for what sets n collapses from all microstructure to frictional
# microstructure.
#
# DESIGN -- range, not seeds. The error on a log-slope goes as
# sigma_resid / (sqrt(N) * spread in ln Theta), so widening the Theta window
# beats adding seeds run-for-run. The 2D mu_g=0 cell currently spans 12x in
# Theta; the warm extension below should take it past 100x. That also attacks
# the project's deepest structural limitation -- everything so far is measured
# over about one decade of Theta, which is why "n" has only ever been a local
# slope rather than an exponent.
#
# Cold is not available: the coldest existing setpoint is already gated out
# (barostat sigma_P/<P> = 0.54, I high by 16%) because shear heating sets a
# floor. So the extension is entirely warm.
#
# Also adds mu_g = 0.02 in 2D. With mu_g = 0, 0.02, 0.05 resolved, the APPROACH
# to zero can be fitted -- if n ~ mu_g^alpha near the origin, alpha is a
# characterisation of the frictional origin rather than another exclusion.
#
# Everything writes into sweep_steady2d / sweep_steady3d with the existing
# naming, so it pools automatically with the campaign and re-runs nothing that
# is already done.
#
# ~87 new runs, ~5 h at NJOBS=16.
#
# Usage:  ./run_mu0_deep.sh [njobs]
set -u
cd "$(dirname "$0")"
NJOBS=${1:-16}
NEQ=500000; NME=500000; CDUMP=25000; CDUMP3=50000
PCONF=10.0; GDOT=1.0e-3
SEEDS=(1 2 3)

T_OLD=(0.0005 0.001 0.0015 0.003 0.005 0.008 0.013 0.02 0.03)
T_WARM2D=(0.05 0.08 0.12 0.2 0.3)
T_WARM3D=(0.2 0.3 0.5)

D2=sweep_steady2d; D3=sweep_steady3d
for s in "${SEEDS[@]}"; do
  [ -f "$D2/data.m2d_seed${s}" ] || cp "sweep_matched2d/data.m2d_seed${s}" "$D2/data.m2d_seed${s}"
  [ -f "$D3/data.granular3d_seed${s}" ] || cp "sweep_run_3d/data.granular3d_seed${s}" "$D3/data.granular3d_seed${s}"
done

throttle(){ while [ "$(jobs -rp | wc -l)" -ge "$NJOBS" ]; do wait -n 2>/dev/null || sleep 1; done; }
launch(){ local d=$1 lab=$2; shift 2
  if [ -f "$d/log.${lab}" ] && grep -q "^DONE" "$d/log.${lab}"; then return; fi
  rm -f "$d/log.${lab}"
  ( cd "$d" && lmp_serial "$@" -var label "$lab" > "run.${lab}.out" 2>&1 ) &
  throttle; }

# 2D: mu_g = 0 and 0.02, full Theta grid incl. warm extension
for mg in 0.0 0.02; do
  for T in "${T_OLD[@]}" "${T_WARM2D[@]}"; do
    for s in "${SEEDS[@]}"; do
      launch $D2 "mu${mg}_T${T}_s${s}" -in ../in.granular_2d_ss \
        -var datafile "data.m2d_seed${s}" -var mu_g $mg -var gdot $GDOT \
        -var Pconf $PCONF -var Tgran $T -var seed $s \
        -var nequil $NEQ -var nmeas $NME -var cdump_every $CDUMP
done; done; done

# 3D: mu_g = 0, warm extension + third seed
for T in "${T_OLD[@]}" 0.05 0.08 0.12 "${T_WARM3D[@]}"; do
  for s in "${SEEDS[@]}"; do
    launch $D3 "mu0.0_T${T}_s${s}" -in ../in.granular_3d_ss \
      -var datafile "data.granular3d_seed${s}" -var mu_g 0.0 -var gdot $GDOT \
      -var Pconf $PCONF -var Tgran $T -var seed $s \
      -var nequil $NEQ -var nmeas $NME -var cdump_every $CDUMP3
done; done

wait
echo "MU0_DEEP_COMPLETE"
