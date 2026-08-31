#!/bin/bash
# Is n converged in 3D? A direct test, not an extrapolation.
#
# WHY ONLY 3D. Splitting the production measurement window in half and fitting
# n separately on each half gives:
#     2D:  mean shift -0.0012  (RMS 0.0228)  -> no systematic trend, converged
#     3D:  mean shift -0.0092  (RMS 0.0145)  -> negative in 6 of 8 cells
# So 2D needs nothing. In 3D n is still falling with strain at roughly 0.009
# per half-window, which is comparable to the 3D error bars (0.012-0.040) --
# and 3D is where every revision to the constraint list originated. mu itself
# drifts +10% across the window in 3D against +5.7% in 2D, consistent with the
# same picture.
#
# WHY A LONG RUN RATHER THAN A LONGER PRODUCTION SWEEP. Extrapolating "one more
# window" assumes the drift is linear in strain; approach to steady state is
# usually exponential, so that assumption is exactly what needs testing. These
# runs use nmeas = 400000 (5x production), which allows n to be fitted from
# SUCCESSIVE SUB-WINDOWS within a single run. That yields a convergence CURVE
# -- n against accumulated strain -- and answers whether n plateaus, rather
# than only whether it is currently moving.
#
# SCOPE. Three frictions spanning the range (0, 0.3, 1.0) x 6 Theta setpoints
# x 2 seeds = 36 runs at 5x length. Six setpoints is enough for a quadratic
# plus a leave-one-out jackknife. This is a validity check, not a replacement
# sweep: if n plateaus at the production value the existing numbers stand as
# they are; if it plateaus lower, the offset is a systematic to apply to 3D.
#
# Queued to start after run_iscan.sh finishes so the two do not oversubscribe
# the machine.
set -u
cd "$(dirname "$0")"
NJOBS=${1:-14}
PCONF=10.0; GDOT=1.0e-3
mkdir -p sweep_3dconv
for s in 1 2; do
  [ -f "sweep_3dconv/data.granular3d_seed${s}" ] || cp "sweep_run_3d/data.granular3d_seed${s}" "sweep_3dconv/data.granular3d_seed${s}"
done

launch(){ local lab=$1; shift
  [ -f "sweep_3dconv/log.${lab}" ] && grep -q "^DONE" "sweep_3dconv/log.${lab}" && return
  ( cd sweep_3dconv && lmp_serial "$@" -var label "$lab" > "run.${lab}.out" 2>&1 ) &
  while [ "$(jobs -rp | wc -l)" -ge "$NJOBS" ]; do wait -n 2>/dev/null || sleep 1; done
}

for mg in 0.0 0.3 1.0; do
 for T in 0.001 0.003 0.008 0.02 0.06 0.12; do
  for s in 1 2; do
   launch "mu${mg}_T${T}_s${s}" -in ../in.granular_3d \
     -var datafile "data.granular3d_seed${s}" -var mu_g $mg -var gdot $GDOT \
     -var Pconf $PCONF -var Tgran $T -var seed $s \
     -var nequil 40000 -var nmeas 400000
done; done; done

wait
echo "3D_CONVERGE_COMPLETE"
