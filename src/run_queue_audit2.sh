#!/bin/bash
# Post-audit queue.  Two campaigns, one shared job pool.
#
#  A. sweep_nothermo2 -- the ATHERMAL CONTROL.  Answers the single largest
#     scope objection to the paper: every Theta in the project is 10^2-10^4x
#     the shear-generated m(gdot d)^2, so a referee can say the whole scan
#     sits outside the domain where mu*Theta^n = F(I) is asserted.  Here the
#     Langevin bath is removed entirely and Theta settles where shear heating
#     balances inelastic dissipation.  Restitution e scans Theta at fixed I
#     with no bath anywhere, so the athermal (Theta, mu) points can be
#     overlaid on the thermostatted sweep_restit at MATCHED e, mu_g, gdot and
#     packing (data.re_seed*).  One curve = the protocol is validated; two
#     curves = the central quantity is protocol-dependent.  Cheap, and it
#     lands first.
#
#  B. sweep_iscan3d -- the 3D INERTIAL-NUMBER SCAN.  The separability failure
#     (the paper's title claim) is currently 2D-only: every 3D run in the
#     project sits at a single shear rate, I = 3.24e-4.  For a relation whose
#     exponent is claimed to be dimension-dependent, that is the first thing a
#     referee asks for.  Two outer arms a decade apart, sharing sweep_steady3d's
#     packings so the mid arm is the existing campaign.
#
# STRAIN, NOT STEPS -- as in run_iscan_steady.sh.  nequil = nmeas = strain 1.0.
# cdump_every = nequil/10 so every run yields ~21 frames (10 in the measurement
# window), matching sweep_steady3d and keeping extract/tail_frames valid.
#
# Resumable: a run whose log already ends in DONE is skipped.
# Usage:  ./run_queue_audit2.sh [njobs]
set -u
cd "$(dirname "$0")"
NJOBS=${1:-14}
# WAIT FOR AC POWER
# On battery, macOS suspends the background coalition: every lmp_serial sits in
# state T accruing no CPU, with no error in any log, and a suspended run looks
# exactly like a slow one. That cost an hour of misdiagnosis on 2026-08-25.
# Check with:  ps -o state,command | grep lmp_serial
while ! pmset -g ps 2>/dev/null | grep -q "AC Power"; do
  echo "$(date '+%H:%M:%S')  on battery -- waiting for AC before starting" \
       "($(pmset -g batt 2>/dev/null | tail -1 | awk '{print $3}' | tr -d ';'))"
  sleep 60
done
# Only NOW hold the machine awake. Doing this before the AC wait would keep it
# from sleeping while idling on battery and flatten the pack before the charger
# ever arrives. `pmset -g custom` has sleep = 1 minute on AC, and the campaign
# survived 2026-08-25 only because something else held an assertion.
caffeinate -dimsu -w $$ &
disown $!   # BUG FIXED 2026-08-28: without this, bash's bare `wait` below waits on
            # every backgrounded job the script started, INCLUDING this caffeinate --
            # which only exits when the script does. Deadlock: all lmp_serial
            # jobs finished, the driver sat alive with no children for 5+ hours.
            # disown removes it from the job table wait watches, without touching
            # -w $$, so it still releases the wake assertion when the script exits.
echo "$(date '+%H:%M:%S')  on AC power -- starting $NJOBS parallel jobs (wake assertion held)"
echo "  if you unplug mid-run the jobs will silently stall; re-running this"
echo "  script is safe and resumes from wherever it stopped."

throttle(){ while [ "$(jobs -rp | wc -l)" -ge "$NJOBS" ]; do wait -n 2>/dev/null || sleep 1; done; }
launch(){ local d=$1 lab=$2; shift 2
  if [ -f "$d/log.${lab}" ] && grep -q "^DONE" "$d/log.${lab}"; then return; fi
  rm -f "$d/log.${lab}"
  # </dev/null matters: without it these inherit the launching terminal's
  # stdin, and a background process group that touches the terminal is sent
  # SIGTTIN and STOPPED. The first attempt at this queue sat in state T for
  # seven minutes having done twenty seconds of work, silently.
  ( cd "$d" && lmp_serial "$@" -var label "$lab" </dev/null > "run.${lab}.out" 2>&1 ) &
  throttle; }

# ------------------------------------------------- A. athermal control (2D)
DA=sweep_nothermo2; mkdir -p $DA
for s in 1 2; do
  [ -f "$DA/data.re_seed${s}" ] || cp sweep_restit/data.re_seed${s} "$DA/data.re_seed${s}"
done
# gdot=1e-3 matches sweep_restit and sweep_steady2d exactly -> direct overlay
for e in 0.1 0.5 0.9; do for mg in 0.0 0.15 0.3; do for s in 1 2; do
  launch $DA "nt_g1.0e-3_e${e}_mu${mg}_s${s}" -in ../in.granular_2d_nothermo_ss \
    -var datafile "data.re_seed${s}" -var mu_g $mg -var gdot 1.0e-3 \
    -var Pconf 10.0 -var e_rest $e -var seed $s \
    -var nequil 1000000 -var nmeas 1000000
done; done; done
# the two outer I arms at e=0.5, to locate Theta_shear at each I of the 2D I-scan
for mg in 0.0 0.15 0.3; do for s in 1 2; do
  launch $DA "nt_g3.162e-3_e0.5_mu${mg}_s${s}" -in ../in.granular_2d_nothermo_ss \
    -var datafile "data.re_seed${s}" -var mu_g $mg -var gdot 3.162e-3 \
    -var Pconf 10.0 -var e_rest 0.5 -var seed $s \
    -var nequil 316255 -var nmeas 316255
done; done
# kick check: one cell at 10x the starting kick. The steady state must not
# remember it, and that has to be shown, not assumed (see the deck).
for s in 1 2; do
  launch $DA "ntk_g1.0e-3_e0.5_mu0.0_s${s}" -in ../in.granular_2d_nothermo_ss \
    -var datafile "data.re_seed${s}" -var mu_g 0.0 -var gdot 1.0e-3 \
    -var Pconf 10.0 -var e_rest 0.5 -var seed $s -var t_kick 1.0e-3 \
    -var nequil 1000000 -var nmeas 1000000
done
for s in 1 2; do
  launch $DA "nt_g3.162e-4_e0.5_mu0.3_s${s}" -in ../in.granular_2d_nothermo_ss \
    -var datafile "data.re_seed${s}" -var mu_g 0.3 -var gdot 3.162e-4 \
    -var Pconf 10.0 -var e_rest 0.5 -var seed $s \
    -var nequil 3162555 -var nmeas 3162555
done

# ------------------------------------------------------ B. 3D I-scan
DB=sweep_iscan3d; mkdir -p $DB
for s in 1 2; do
  [ -f "$DB/data.granular3d_seed${s}" ] || cp sweep_steady3d/data.granular3d_seed${s} "$DB/data.granular3d_seed${s}"
done
T3D=(0.0005 0.001 0.0015 0.003 0.005 0.008 0.013 0.02 0.03 0.05 0.08 0.12)
# slow arm first: it is the critical path (1.8 h/run vs 0.18 h)
for G in 3.162e-4 3.162e-3; do
  case $G in
    3.162e-4) NS=1581278; CD=158128 ;;
    3.162e-3) NS=158128;  CD=15813  ;;
  esac
  for mg in 0.0 0.15 0.3; do for T in "${T3D[@]}"; do for s in 1 2; do
    launch $DB "g${G}_mu${mg}_T${T}_s${s}" -in ../in.granular_3d_ss \
      -var datafile "data.granular3d_seed${s}" -var mu_g $mg -var gdot $G \
      -var Pconf 10.0 -var Tgran $T -var seed $s \
      -var nequil $NS -var nmeas $NS -var cdump_every $CD
done; done; done; done
# --------------------------------------------- C. velocity-profile check (2D)
# compute temp/deform subtracts the AFFINE streaming field and calls the rest
# Theta. Nothing in the project has ever checked that the profile IS affine,
# and no sweep dumps per-atom velocities, so it cannot be checked on existing
# data. Twelve cells, otherwise identical to sweep_steady2d. ~7 core-hours.
DC=sweep_velprof; mkdir -p $DC
for s in 1 2; do
  [ -f "$DC/data.m2d_seed${s}" ] || cp sweep_steady2d/data.m2d_seed${s} "$DC/data.m2d_seed${s}"
done
for mg in 0.0 0.15 0.3; do for T in 0.001 0.03; do for s in 1 2; do
  launch $DC "mu${mg}_T${T}_s${s}" -in ../in.granular_2d_ss_prof \
    -var datafile "data.m2d_seed${s}" -var mu_g $mg -var gdot 1.0e-3 \
    -var Pconf 10.0 -var Tgran $T -var seed $s \
    -var nequil 500000 -var nmeas 500000 -var cdump_every 250000
done; done; done

wait
echo "AUDIT2_QUEUE_COMPLETE"
