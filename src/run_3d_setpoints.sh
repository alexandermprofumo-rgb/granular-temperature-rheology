#!/bin/bash
# Extend the 3D Theta grid warm. This is the ONE remaining place where more
# simulation clearly buys resolution.
#
# WHY. Every revision in the audited constraint list (§6) was driven by 3D. The
# quadratic in log-log is inadequate there -- an F-test finds a required cubic
# at p < 0.003 at EVERY mu_g >= 0.2 -- and with only 9 setpoints the fit cannot
# support the model the data demand. The resulting window systematic runs
# 2.3-2.9x the statistical error in 3D against 0.2-1.7x in 2D. That is what
# withdrew constraint 5 (A_3/A_2 fell to 1.4 sigma), left both peak locations
# undetermined, and inflated constraints 1, 3 and 4.
#
# SETPOINTS, NOT SEEDS. The limitation is model inadequacy, not scatter. More
# seeds shrink the statistical term, which is already the smaller one; only
# more Theta setpoints let a cubic be constrained and jackknifed. So this adds
# three setpoints at the existing three seeds -- 72 runs -- rather than a
# larger campaign of extra seeds.
#
# WARM, NOT COLD. Of the 9 existing setpoints, 8-9 survive; the only casualty
# is the coldest (T = 0.0005), which fails fixed-I in several cells. Colder
# points would fail the same way. At the warm end the thermostat stays firmly
# in control (Theta/Tgran = 0.03-0.17, far below the gate at 1) and I is
# steady, so T = 0.06, 0.12, 0.25 should all be usable -- but the gates decide,
# not this comment.
#
# 12 setpoints x 3 seeds = 36 runs per cell, enough to fit a cubic and drop one
# setpoint at a time for the jackknife.
set -u
cd "$(dirname "$0")"
NJOBS=${1:-14}
PCONF=10.0; GDOT=1.0e-3

launch(){ local lab=$1; shift
  [ -f "sweep_run_3d/log.${lab}" ] && grep -q "^DONE" "sweep_run_3d/log.${lab}" && return
  ( cd sweep_run_3d && lmp_serial "$@" -var label "$lab" > "run.${lab}.out" 2>&1 ) &
  while [ "$(jobs -rp | wc -l)" -ge "$NJOBS" ]; do wait -n 2>/dev/null || sleep 1; done
}

for mg in 0.0 0.05 0.1 0.15 0.2 0.3 0.5 1.0; do
 for T in 0.06 0.12 0.25; do
  for s in 1 2 3; do
   launch "mu${mg}_T${T}_s${s}" -in ../in.granular_3d \
     -var datafile "data.granular3d_seed${s}" -var mu_g $mg -var gdot $GDOT \
     -var Pconf $PCONF -var Tgran $T -var seed $s \
     -var nequil 40000 -var nmeas 80000
done; done; done

wait
echo "3D_SETPOINTS_COMPLETE"
