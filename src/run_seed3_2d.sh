#!/bin/bash
# A THIRD SEED FOR THE NINE FRICTIONAL 2D CELLS.
#
# WHY. mu_g = 0 and 0.02 already have three seeds (42 runs each), so the
# headline cells -- constraints 1, 2 and 2b -- are NOT two-seed results. The
# nine frictional 2D cells are, and they carry constraints 3, 5, 6 and the
# dimension comparison.
#
# WHAT A THIRD SEED BUYS, measured on the two cells that already have one:
#
#     mu_g = 0     2 seeds 0.0298 +/- 0.0098  ->  3 seeds 0.0276 +/- 0.0100
#     mu_g = 0.02  2 seeds 0.0547 +/- 0.0093  ->  3 seeds 0.0589 +/- 0.0075
#
# Central values move 0.2-0.5 sigma; the error falls 19% in one cell and RISES
# 2% in the other. That second case is the point: a variance estimated from two
# seeds has ONE degree of freedom, so its own relative uncertainty is 141%
# (sqrt(2/nu)). Three seeds takes that to 100%. The gain is knowing whether the
# error bar is honest, not making it smaller -- and with two seeds a bad run
# (transient unjamming, a barostat excursion) is invisible, because it only
# shifts the mean and there is no third value to arbitrate.
#
# COST. 9 cells x 9 setpoints = 81 runs at ~0.37 h each = ~30 core-hours,
#
# CAUTION. This MOVES every central value in these nine cells: the peak
# positions, the rise/fall significances, constraints 3/4/5. Run it together
# with whichever window/degree policy is chosen for the constraint list, so
# the numbers settle once rather than twice.
#
# Parameters are copied from run_steady_campaign.sh and verified against an
# existing seed-3 run:
#   DONE label=mu0.02_T0.001_s3 mu_g=0.02 gdot=1.0e-3 Tgran=0.001 seed=3
set -u
cd "$(dirname "$0")"
NJOBS=${1:-12}

D=sweep_steady2d
GDOT=1.0e-3
PCONF=10.0
NEQ=500000        # strain 0.5
NME=500000        # strain 0.5
CDUMP=25000
SEED=3

# the nine cells that do not already have a third seed
MU=(0.05 0.1 0.12 0.15 0.17 0.2 0.3 0.5 1.0)
TG=(0.0005 0.001 0.0015 0.003 0.005 0.008 0.013 0.02 0.03)

[ -f "$D/data.m2d_seed${SEED}" ] || cp "sweep_matched2d/data.m2d_seed${SEED}" \
                                      "$D/data.m2d_seed${SEED}"

# WAIT FOR THE MACHINE: AC power, and nothing else running
# Two conditions, not one. AC because guard_power.sh SIGSTOPs lmp_serial off
# mains and a suspended run is indistinguishable from a slow one (state T,
# etime running ahead of time). Runs idle-priority.
#
# Deliberately NOT chained to a log string. queue_bigN.sh polled
# queued_campaigns.log for "ALL COMPLETE", the string appeared, and the watcher
# never fired -- costing a whole night. A condition you can verify by hand is
# safer than a message you have to trust.
while true; do
  on_ac=0; idle=0
  pmset -g ps 2>/dev/null | grep -q "AC Power" && on_ac=1
  [ "$(pgrep -x lmp_serial | wc -l | tr -d ' ')" = "0" ] && idle=1
  [ "$on_ac" = "1" ] && [ "$idle" = "1" ] && break
  echo "$(date '+%H:%M:%S')  waiting -- AC:$on_ac idle:$idle" \
       "($(pmset -g batt 2>/dev/null | tail -1 | awk '{print $3}' | tr -d ';'))"
  sleep 60
done
caffeinate -dimsu -w $$ &
disown $!   # BUG FIXED 2026-08-28: without this, bash's bare `wait` below waits on
            # every backgrounded job the script started, INCLUDING this caffeinate --
            # which only exits when the script does. Deadlock: all lmp_serial
            # jobs finished, the driver sat alive with no children for 5+ hours.
            # disown removes it from the job table wait watches, without touching
            # -w $$, so it still releases the wake assertion when the script exits.
echo "$(date '+%F %T')  starting third-seed campaign, $NJOBS jobs, 81 runs"

throttle(){ while [ "$(jobs -rp | wc -l)" -ge "$NJOBS" ]; do wait -n 2>/dev/null || sleep 1; done; }

launch(){ local lab=$1; shift
  # resumable: a finished run is never repeated
  if [ -f "$D/log.${lab}" ] && grep -q "^DONE" "$D/log.${lab}"; then return; fi
  rm -f "$D/log.${lab}"
  # </dev/null: without it a background job that touches the terminal takes
  # SIGTTIN and stops silently.
  ( cd "$D" && nice -n 5 lmp_serial "$@" -var label "$lab" </dev/null \
      > "run.${lab}.out" 2>&1 ) &
  throttle; }

for mg in "${MU[@]}"; do for T in "${TG[@]}"; do
  launch "mu${mg}_T${T}_s${SEED}" -in ../in.granular_2d_ss \
    -var datafile "data.m2d_seed${SEED}" -var mu_g $mg -var gdot $GDOT \
    -var Pconf $PCONF -var Tgran $T -var seed $SEED \
    -var nequil $NEQ -var nmeas $NME -var cdump_every $CDUMP
done; done
wait
echo "$(date '+%F %T')  SEED3_2D_COMPLETE  ($(grep -l '^DONE' $D/log.*_s3 2>/dev/null | wc -l | tr -d ' ') seed-3 runs on disk)"
echo "Next: re-run constraints_final.py and analyze_channel_state.py -- every"
echo "central value in the nine frictional cells will have moved."
