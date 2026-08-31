#!/bin/bash
# TIER 1b: the LEVERS at strain 1.0.
#
# The steady campaign re-founded constraints 1-6. Constraints 7-10, all nine
# two-lever tests, section 7a's chi falsification, section 9's pressure
# response and section 12's crossings still rest on strain-0.12 runs -- and
# they are the results MOST exposed, not the least:
#
#   * the transient bias is friction-dependent, and it is also stiffness- and
#     pressure-dependent, because those set how fast the contact network
#     reorganises. A two-lever test compares n at two (E, P, k_t) states each
#     measured at a different fraction of its own convergence, so a
#     "discrepancy" can be pure protocol mismatch.
#   * constraint 9's b = dn/dln(kappa) is literally a derivative with respect
#     to stiffness -- the single quantity most vulnerable to a
#     stiffness-dependent transient.
#   * section 12's crossings are friction-axis locations where a derivative
#     changes sign, sitting on top of a friction-dependent bias.
#
# STRAIN, NOT STEPS. Each deck has its own dt (the stiffness deck scales it as
# E^-2/5; the pressure deck as 1/sqrt(P)), and strain per step is gdot*dt. Step
# counts are therefore computed per cell as strain/(gdot*dt), so every run in
# every arm reaches the SAME strain -- 0.5 equilibration + 0.5 measurement.
# That is the whole point: comparing levers at matched step count is what
# created the problem.
#
# All decks now dump the tangential force, so a_t and chi are measured on the
# same runs that give n -- section 7a's separate 432-run chi campaign is no
# longer needed.
#
# STAGED, biggest scientific payload first:
#   pressure  432 runs  ~23 h  -> C9, chi (7a), the two crossings (12)
#   stiffness 270 runs  ~18 h  -> C7, C8
#   kt        216 runs  ~11 h  -> C7
#   pidamp    324 runs  ~17 h  -> C10, on a Theta window 3x wider than before
#                        ~69 h total at NJOBS=16
#
# Usage:  ./run_levers_steady.sh [njobs] [pressure|stiffness|kt|pidamp|all]
set -u
cd "$(dirname "$0")"
NJOBS=${1:-16}
STAGE=${2:-all}
CDUMP=25000
SEEDS=(1 2)
STRAIN_EQ=0.5
STRAIN_ME=0.5

throttle(){ while [ "$(jobs -rp | wc -l)" -ge "$NJOBS" ]; do wait -n 2>/dev/null || sleep 1; done; }
launch(){ local d=$1 lab=$2; shift 2
  if [ -f "$d/log.${lab}" ] && grep -q "^DONE" "$d/log.${lab}"; then return; fi
  rm -f "$d/log.${lab}"
  # nice: the campaign is background work. Leaving the OS and anything
  # interactive at higher priority is also part of why the machine went down --
  # it slept with 16 CPU-bound jobs saturating every core and the wake
  # transition could not quiesce them in time, so the watchdog forced a restart.
  ( cd "$d" && nice -n 5 lmp_serial "$@" -var label "$lab" > "run.${lab}.out" 2>&1 ) &
  throttle; }

# steps for a given strain, from gdot and dt
# Python interpreter.  Override with  PY=/path/to/python  in the environment;
# otherwise prefer the project virtualenv, then fall back to python3 on PATH.
if [ -z "${PY:-}" ]; then
  _venv="$(cd "$(dirname "$0")/.." 2>/dev/null && pwd)/.venv/bin/python"
  if [ -x "$_venv" ]; then PY="$_venv"; else PY=python3; fi
fi
steps(){ "$PY" -c \
  "import sys;print(int(round(float(sys.argv[1])/(float(sys.argv[2])*float(sys.argv[3])))))" "$1" "$2" "$3"; }

TG=(0.0005 0.001 0.0015 0.003 0.005 0.008 0.013 0.02 0.03)

# ---------------------------------------------------------------- pressure
do_pressure(){
  local D=sweep_lev_P; mkdir -p $D
  for s in "${SEEDS[@]}"; do
    [ -f "$D/data.pi_seed${s}" ] || cp sweep_pichi/data.pi_seed${s} "$D/data.pi_seed${s}"
  done
  for P in 5.0 10.0 25.0 50.0; do
    read gdot tdamp dt < <("$PY" -c \
      "import math;P=$P;print(f'{3.162e-4*math.sqrt(P):.6e} {1.581/math.sqrt(P):.6f} {0.001*math.sqrt(10.0/P):.6e}')")
    NEQ=$(steps $STRAIN_EQ $gdot $dt); NME=$(steps $STRAIN_ME $gdot $dt)
    for mg in 0.05 0.1 0.15 0.2 0.3 0.5; do
      for T10 in "${TG[@]}"; do
        Tg=$("$PY" -c "print($T10*$P/10.0)")
        for s in "${SEEDS[@]}"; do
          launch $D "P${P}_mu${mg}_T${T10}_s${s}" -in ../in.granular_2d_piscan_ss \
            -var datafile "data.pi_seed${s}" -var mu_g $mg -var Pconf $P \
            -var gdot $gdot -var tdamp $tdamp -var dt $dt -var Tgran $Tg \
            -var seed $s -var nequil $NEQ -var nmeas $NME -var cdump_every $CDUMP
  done; done; done; done; }

# --------------------------------------------------------------- stiffness
do_stiffness(){
  local D=sweep_lev_E; mkdir -p $D
  for s in "${SEEDS[@]}"; do
    [ -f "$D/data.stiff_seed${s}" ] || cp sweep_stiffness/data.stiff_seed${s} "$D/data.stiff_seed${s}"
  done
  for E in 1.0e4 3.0e4 1.0e5 3.0e5 1.0e6; do
    read dt nr < <("$PY" in.granular_2d_stiff.py "$E" | awk '{print $1, $2}')
    NEQ=$(steps $STRAIN_EQ 1.0e-3 $dt); NME=$(steps $STRAIN_ME 1.0e-3 $dt)
    for mg in 0.1 0.3 1.0; do for T in "${TG[@]}"; do for s in "${SEEDS[@]}"; do
      launch $D "E${E}_mu${mg}_T${T}_s${s}" -in ../in.granular_2d_stiff_ss \
        -var datafile "data.stiff_seed${s}" -var mu_g $mg -var gdot 1.0e-3 \
        -var Pconf 10.0 -var Tgran $T -var seed $s -var Emod $E -var dt $dt \
        -var nrelax $nr -var nequil $NEQ -var nmeas $NME -var cdump_every $CDUMP
  done; done; done; done; }

# ---------------------------------------------------------------------- kt
do_kt(){
  local D=sweep_lev_kt; mkdir -p $D
  for s in "${SEEDS[@]}"; do
    [ -f "$D/data.kt_seed${s}" ] || cp sweep_kt_P10/data.ktP10_seed${s} "$D/data.kt_seed${s}"
  done
  NEQ=$(steps $STRAIN_EQ 1.0e-3 0.001); NME=$(steps $STRAIN_ME 1.0e-3 0.001)
  for kt in 2.0e4 5.0e4 9.05e4 2.0e5; do for mg in 0.1 0.3 1.0; do
    for T in "${TG[@]}"; do for s in "${SEEDS[@]}"; do
      launch $D "k${kt}_mu${mg}_T${T}_s${s}" -in ../in.granular_2d_sens_ss \
        -var datafile "data.kt_seed${s}" -var mu_g $mg -var gdot 1.0e-3 \
        -var Pconf 10.0 -var Tgran $T -var seed $s -var kt_spec $kt \
        -var nequil $NEQ -var nmeas $NME -var cdump_every $CDUMP
  done; done; done; done; }

# ------------------------------------------------------------------ pidamp
# Theta grid extended warm: constraint 10's window was only 1.9-2.8x, which is
# why a null there could never be worth much. A null is worth exactly its power.
do_pidamp(){
  local D=sweep_lev_td; mkdir -p $D
  for s in "${SEEDS[@]}"; do
    [ -f "$D/data.pd_seed${s}" ] || cp sweep_pidamp/data.pd_seed${s} "$D/data.pd_seed${s}"
  done
  NEQ=$(steps $STRAIN_EQ 1.0e-3 0.001); NME=$(steps $STRAIN_ME 1.0e-3 0.001)
  for td in 0.125 0.25 0.5 1.0 2.0 4.0; do for mg in 0.1 0.3 1.0; do
    for T in "${TG[@]}" 0.05 0.08; do for s in "${SEEDS[@]}"; do
      launch $D "td${td}_mu${mg}_T${T}_s${s}" -in ../in.granular_2d_piscan_ss \
        -var datafile "data.pd_seed${s}" -var mu_g $mg -var Pconf 10.0 \
        -var gdot 1.0e-3 -var tdamp $td -var dt 0.001 -var Tgran $T \
        -var seed $s -var nequil $NEQ -var nmeas $NME -var cdump_every $CDUMP
  done; done; done; done; }

case "$STAGE" in
  pressure)  do_pressure ;;
  stiffness) do_stiffness ;;
  kt)        do_kt ;;
  pidamp)    do_pidamp ;;
  all)       do_pressure; wait; do_stiffness; wait; do_kt; wait; do_pidamp ;;
  *) echo "unknown stage: $STAGE"; exit 1 ;;
esac
wait
echo "LEVERS_STEADY_COMPLETE stage=$STAGE"
