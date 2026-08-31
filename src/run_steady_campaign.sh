#!/bin/bash
# TIER 1: the steady-state re-run.
#
# WHY. Every production sweep in this project shears to total strain
# gdot*dt*(nequil+nmeas) = 0.12. Measured across all of them, mu is still
# rising +14.9% (2D) / +23.9% (3D) between the first and second halves of the
# measurement window, in 97-99% of runs. That alone would be survivable -- a
# Theta-independent drift cancels in a log slope -- but it is NOT
# Theta-independent. Against the strain-1.5 control:
#
#     mu(production window) / mu(steady), by setpoint
#     mu_g = 0.1 :  0.936  0.942  0.921  0.941   -> flat  -> bias on n +0.000
#     mu_g = 0.3 :  0.990  0.957  0.946  0.920   -> grows -> bias on n +0.029
#
# and +0.029 is the entire gap between the production n = 0.145 and the
# strain-1.5 value 0.116 at mu_g = 0.3. Hotter samples are further behind in
# building their steady-state strength, so part of what has been reported as
# "agitation weakens the material" is "the hot cells have not finished getting
# strong yet". The bias is ~0 at low friction and largest at intermediate
# friction -- exactly where the peak of n(mu_g) lives.
#
# The decks also now dump the TANGENTIAL contact force (p1,p2,p4), so a_t and
# chi are measured rather than closed as 2*mu - a_c - a_n. That retires the
# closure caveat in 2D and the calibrated-prefactor hack in 3D at the same
# time, on the same runs.
#
# SIZING -- measured, not guessed.
#   Convergence strain: a_c, a_n and mu all reach 90% of their steady values by
#   strain 0.06 and are flat from ~0.2 (301 dump frames of a strain-1.5 run).
#   nequil = 5e5 is strain 0.5, an 8x margin. nmeas = 5e5 is strain 0.5, which
#   gave stat ~ 0.004-0.006 on n in the strain-1.5 control against 0.010-0.020
#   in production. Two seeds, not four: at strain 0.5 the initial packing is
#   forgotten, so seeds no longer carry the weight they did at strain 0.12.
#
#   Cost: 2D 180 runs + 3D 240 runs, 1e6 steps each, ~124 CPU-hr.
#
# SELF-CHECK. check_steady_campaign.py refits n on the first and second half of
# each measurement window. A cell whose halves disagree is not converged and
# should be re-run longer INDIVIDUALLY -- do not pad every run to cover the
# worst one.
#
# STAGING. Run the pilot first (./run_steady_campaign.sh 16 pilot): four 2D
# frictions, ~1 h, enough to size the correction before committing the rest.
#
# Usage:  ./run_steady_campaign.sh [njobs] [pilot|2d|3d|all]
set -u
cd "$(dirname "$0")"
NJOBS=${1:-16}
STAGE=${2:-all}

NEQ=500000        # strain 0.5
NME=500000        # strain 0.5
CDUMP=25000       # 2D: 40 frames over the run, 20 inside the measurement window
# 3D dumps 14 columns over ~21k contacts, ~3x the 2D volume per frame. Half the
# frame rate: 20 frames total, of which the last 8 (which is all analyze_rb_3d
# and analyze_exact_split use) span steps 650k-1000k, comfortably inside the
# measurement window that opens at 500k. Saves ~10 GB for nothing.
CDUMP3=50000
PCONF=10.0
GDOT=1.0e-3
SEEDS=(1 2)

MU2D=(0.0 0.05 0.1 0.12 0.15 0.17 0.2 0.3 0.5 1.0)
T2D=(0.0005 0.001 0.0015 0.003 0.005 0.008 0.013 0.02 0.03)
MU3D=(0.0 0.05 0.1 0.15 0.2 0.3 0.4 0.5 0.6 1.0)
T3D=(0.0005 0.001 0.0015 0.003 0.005 0.008 0.013 0.02 0.03 0.05 0.08 0.12)

MUPILOT=(0.0 0.1 0.2 0.3)

D2=sweep_steady2d
D3=sweep_steady3d
mkdir -p $D2 $D3
for s in "${SEEDS[@]}"; do
  [ -f "$D2/data.m2d_seed${s}" ] || cp "sweep_matched2d/data.m2d_seed${s}" "$D2/data.m2d_seed${s}"
  [ -f "$D3/data.granular3d_seed${s}" ] || cp "sweep_run_3d/data.granular3d_seed${s}" "$D3/data.granular3d_seed${s}"
done

throttle(){ while [ "$(jobs -rp | wc -l)" -ge "$NJOBS" ]; do wait -n 2>/dev/null || sleep 1; done; }

launch(){ local d=$1 lab=$2; shift 2
  # resumable: a finished run is never repeated
  if [ -f "$d/log.${lab}" ] && grep -q "^DONE" "$d/log.${lab}"; then return; fi
  rm -f "$d/log.${lab}"          # a partial log would confuse parse_log's append
  ( cd "$d" && lmp_serial "$@" -var label "$lab" > "run.${lab}.out" 2>&1 ) &
  throttle
}

run2d(){ local mus=("$@")
  for mg in "${mus[@]}"; do for T in "${T2D[@]}"; do for s in "${SEEDS[@]}"; do
    launch $D2 "mu${mg}_T${T}_s${s}" -in ../in.granular_2d_ss \
      -var datafile "data.m2d_seed${s}" -var mu_g $mg -var gdot $GDOT \
      -var Pconf $PCONF -var Tgran $T -var seed $s \
      -var nequil $NEQ -var nmeas $NME -var cdump_every $CDUMP
  done; done; done; }

run3d(){ local mus=("$@")
  for mg in "${mus[@]}"; do for T in "${T3D[@]}"; do for s in "${SEEDS[@]}"; do
    launch $D3 "mu${mg}_T${T}_s${s}" -in ../in.granular_3d_ss \
      -var datafile "data.granular3d_seed${s}" -var mu_g $mg -var gdot $GDOT \
      -var Pconf $PCONF -var Tgran $T -var seed $s \
      -var nequil $NEQ -var nmeas $NME -var cdump_every $CDUMP3
  done; done; done; }

case "$STAGE" in
  pilot) echo "PILOT: 2D, frictions ${MUPILOT[*]}"; run2d "${MUPILOT[@]}" ;;
  2d)    run2d "${MU2D[@]}" ;;
  3d)    run3d "${MU3D[@]}" ;;
  all)   run2d "${MU2D[@]}"; run3d "${MU3D[@]}" ;;
  *)     echo "unknown stage: $STAGE"; exit 1 ;;
esac

wait
echo "STEADY_CAMPAIGN_COMPLETE stage=$STAGE"
