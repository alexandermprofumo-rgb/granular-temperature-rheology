#!/bin/bash
# THE INERTIAL-NUMBER SCAN AT STRAIN 1.0  (section 11, and F(I) itself).
#
# The old scan was strain-0.12 and gave chi2/dof = 3.6 for a common
# dn/dln(I), so the headline "-0.0320 +/- 0.0056, 5.7 sigma" was a weighted
# mean over values inconsistent with each other.
#
# STRAIN, NOT STEPS. nsteps = 0.5/(gdot*dt) per phase, so the slow arm needs
# 1,581,278 steps per phase against 158,128 for the fast one -- that asymmetry
# is the whole cost of the experiment, and it is unavoidable if the arms are to
# be compared at equal strain.
#
# The MIDDLE arm (gdot = 1e-3) already exists in sweep_steady2d at all four
# frictions, so only the outer two are run here.
#
# Measuring mu(Theta, I) on this grid IS F(I) -- the relation under test has
# two sides and only the Theta side has ever been examined.
set -u
cd "$(dirname "$0")"
NJOBS=${1:-12}
D=sweep_iscan2; mkdir -p $D
CDUMP=25000
for s in 1 2; do
  [ -f "$D/data.m2d_seed${s}" ] || cp sweep_matched2d/data.m2d_seed${s} "$D/data.m2d_seed${s}"
done
TG=(0.0005 0.001 0.0015 0.003 0.005 0.008 0.013 0.02 0.03 0.05 0.08)

throttle(){ while [ "$(jobs -rp | wc -l)" -ge "$NJOBS" ]; do wait -n 2>/dev/null || sleep 1; done; }
launch(){ local lab=$1; shift
  if [ -f "$D/log.${lab}" ] && grep -q "^DONE" "$D/log.${lab}"; then return; fi
  rm -f "$D/log.${lab}"
  ( cd "$D" && nice -n 5 lmp_serial "$@" -var label "$lab" > "run.${lab}.out" 2>&1 ) &
  throttle; }

# fast arm first: 10x cheaper, so a usable result lands early
for G in 3.162e-3 3.162e-4; do
  case $G in
    3.162e-3) NS=158128 ;;
    3.162e-4) NS=1581278 ;;
  esac
  for mg in 0.0 0.15 0.3 1.0; do for T in "${TG[@]}"; do for s in 1 2; do
    launch "g${G}_mu${mg}_T${T}_s${s}" -in ../in.granular_2d_ss \
      -var datafile "data.m2d_seed${s}" -var mu_g $mg -var gdot $G \
      -var Pconf 10.0 -var Tgran $T -var seed $s \
      -var nequil $NS -var nmeas $NS -var cdump_every $CDUMP
done; done; done; done
wait
echo "ISCAN_STEADY_COMPLETE"
