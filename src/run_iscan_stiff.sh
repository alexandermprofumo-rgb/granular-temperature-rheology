#!/bin/bash
# ---------------------------------------------------------------------------
# THE CONFOUND CAMPAIGN: the three shear-rate arms repeated at ten times the
# modulus.
#
# WHY. Along an arm, E and P are fixed and gdot varies tenfold, so the inertial
# number I and the contact time relative to the shear time, t_c*gdot, move
# together by exactly the same factor. Every number in Sec. IV is measured
# along those arms, so Sec. IV cannot distinguish "n depends on I" from "n
# depends on t_c*gdot". The modulus lever runs at one gdot and the arms run at
# one E: two one-dimensional slices crossing at a single point. From a cross
# you can measure both partial derivatives at the crossing and nothing else.
#
# confound_test.py showed the consequence. At mu_g = 0.3 the arms fall on the
# curve the modulus lever traces, RMS 1.3 sigma, so a single dependence on
# t_c*gdot reproduces both results. The two slopes differ by only 1.7 sigma
# under t_c ~ E^-2/5 and 2.6 sigma under t_c ~ E^-1/3, and nothing in the
# existing data selects between those exponents.
#
# WHAT THIS ADDS. A second row of the grid. At E = 1e6 the whole arm family
# shifts to t_c*gdot lower by 10^-0.4 = 0.40 while I is unchanged, so:
#
#   if n = n(t_c*gdot)   the shifted arms land on the SAME curve as the
#                        modulus lever, offset from the old arms
#   if n = n(I)          the shifted arms lie on top of the OLD arms and
#                        away from the modulus curve
#
# The two predictions differ by about 0.09*ln(10^0.4) = 0.083 in n, against
# per-cell errors near 0.02. This is a decisive test, not a bound.
#
# MU_G = 0 IS THE CONTROL, and it is not optional. Separability holds there
# (dn/dlnI = 0.0015 +/- 0.0071), and a contact-time effect has no obvious
# reason to switch off without friction. If the frictionless arms move with E
# while their I dependence stays null, t_c*gdot is not what n responds to.
#
# STRAIN, NOT STEPS. The stiffness deck scales dt as E^-2/5, so at E = 1e6 the
# timestep is 2.5x smaller and every arm needs 2.5x the steps for the same
# strain. Step counts are computed per cell from strain/(gdot*dt) so that all
# runs reach 0.5 equilibration plus 0.5 measurement, matching the existing
# arms.
#
# PACKINGS. The same data.m2d_seed files the existing 2D arms use, so the new
# family is paired with the old one configuration by configuration. The mid arm
# is rerun here rather than reused from sweep_lev_E, which was built on
# different packings.
#
# Usage:  ./run_iscan_stiff.sh [njobs]      (default 12)
# Stop:   pkill -f lmp_serial               (partial runs are redone on rerun)
# ---------------------------------------------------------------------------
set -u
cd "$(dirname "$0")"
NJOBS=${1:-12}
D=sweep_iscan_stiff; mkdir -p $D
EMOD=1.0e6
CDUMP=25000
SEEDS=(1 2)
STRAIN_EQ=0.5
STRAIN_ME=0.5
TG=(0.0005 0.001 0.0015 0.003 0.005 0.008 0.013 0.02 0.03 0.05 0.08)

if [ -z "${PY:-}" ]; then
  _venv="$(cd .. && pwd)/.venv/bin/python"
  if [ -x "$_venv" ]; then PY="$_venv"; else PY=python3; fi
fi

for s in "${SEEDS[@]}"; do
  [ -f "$D/data.m2d_seed${s}" ] || cp sweep_matched2d/data.m2d_seed${s} "$D/data.m2d_seed${s}"
done

steps(){ "$PY" -c \
  "import sys;print(int(round(float(sys.argv[1])/(float(sys.argv[2])*float(sys.argv[3])))))" "$1" "$2" "$3"; }

throttle(){ while [ "$(jobs -rp | wc -l)" -ge "$NJOBS" ]; do wait -n 2>/dev/null || sleep 1; done; }
launch(){ local lab=$1; shift
  if [ -f "$D/log.${lab}" ] && grep -q "^DONE" "$D/log.${lab}"; then return; fi
  rm -f "$D/log.${lab}"
  ( cd "$D" && nice -n 5 lmp_serial "$@" -var label "$lab" > "run.${lab}.out" 2>&1 ) &
  throttle; }

read DT NRELAX _ < <("$PY" in.granular_2d_stiff.py "$EMOD")
echo "E = $EMOD   dt = $DT   nrelax = $NRELAX"

# Fast arm first so a usable comparison lands early; the slow arm is 10x the
# cost of the fast one and 25x the cost at baseline stiffness.
for G in 3.162e-3 1.0e-3 3.162e-4; do
  NEQ=$(steps $STRAIN_EQ $G $DT); NME=$(steps $STRAIN_ME $G $DT)
  echo "  gdot = $G   nequil = $NEQ   nmeas = $NME"
  for mg in 0.3 0.0; do for T in "${TG[@]}"; do for s in "${SEEDS[@]}"; do
    launch "g${G}_mu${mg}_T${T}_s${s}" -in ../in.granular_2d_stiff_ss \
      -var datafile "data.m2d_seed${s}" -var mu_g $mg -var gdot $G \
      -var Pconf 10.0 -var Tgran $T -var seed $s -var Emod $EMOD -var dt $DT \
      -var nrelax $NRELAX -var nequil $NEQ -var nmeas $NME -var cdump_every $CDUMP
  done; done; done
done
wait
echo "campaign complete: $(ls $D/log.g* 2>/dev/null | wc -l) logs"
