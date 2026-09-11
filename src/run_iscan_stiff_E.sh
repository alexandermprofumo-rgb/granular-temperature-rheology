#!/bin/bash
# ---------------------------------------------------------------------------
# THE SHEAR-RATE ARMS AT A CHOSEN MODULUS.
#
# Generalizes run_iscan_stiff.sh, which fixed E = 1e6 and cannot be edited
# while it runs (bash reads a script as it executes it).
#
# WHY A THIRD STIFFNESS. confound_grid.py showed that the two live accounts of
# the shear-rate dependence of n are both statements about one function
# n(ln I, ln kappa), because the contact time relative to the shear time is
# itself t_c*gdot = I * kappa^(-alpha):
#
#   A  n = F(ln I - alpha ln kappa)     changing kappa shifts n(ln I) sideways
#   B  n = G(ln I) + H(ln kappa)        changing kappa shifts n(ln I) up
#
# If n is straight in ln I those two shifts are indistinguishable, which is why
# two stiffnesses reached only 1.7 sigma. What separates them is curvature: a
# sideways shift changes the local slope at fixed I and an upward shift does
# not. Under B the arm slope must be the same at every stiffness; under A it
# drifts with kappa. The slope was -0.063 at E = 1e5 and -0.081 at 1e6, a 1.2
# sigma difference. A third stiffness turns that difference into a trend with
# its own error.
#
# A density scan cannot do this. At fixed I, kappa and Pi_damp, changing the
# grain density is a change of units and leaves every dimensionless quantity,
# t_c*gdot included, exactly where it was.
#
# E = 1e4 extends the arms down to kappa = 1e3, two decades in total, and is
# cheap: dt scales as E^-2/5, so steps per unit strain are 6.3x fewer than at
# 1e6.
#
# Usage:  ./run_iscan_stiff_E.sh EMOD DIR [njobs] [mu_g ...]
#         ./run_iscan_stiff_E.sh 1.0e4 sweep_iscan_stiff_E1e4 12 0.3
# ---------------------------------------------------------------------------
set -u
cd "$(dirname "$0")"
EMOD=${1:?modulus}
D=${2:?output directory}
NJOBS=${3:-12}
shift 3 2>/dev/null || shift $#
MUS=("$@"); [ ${#MUS[@]} -eq 0 ] && MUS=(0.3)
mkdir -p "$D"
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
echo "E = $EMOD   dt = $DT   nrelax = $NRELAX   mu_g = ${MUS[*]}   -> $D"

for G in 3.162e-3 1.0e-3 3.162e-4; do
  NEQ=$(steps $STRAIN_EQ $G $DT); NME=$(steps $STRAIN_ME $G $DT)
  echo "  gdot = $G   nequil = $NEQ   nmeas = $NME"
  for mg in "${MUS[@]}"; do for T in "${TG[@]}"; do for s in "${SEEDS[@]}"; do
    launch "g${G}_mu${mg}_T${T}_s${s}" -in ../in.granular_2d_stiff_ss \
      -var datafile "data.m2d_seed${s}" -var mu_g $mg -var gdot $G \
      -var Pconf 10.0 -var Tgran $T -var seed $s -var Emod $EMOD -var dt $DT \
      -var nrelax $NRELAX -var nequil $NEQ -var nmeas $NME -var cdump_every $CDUMP
  done; done; done
done
wait
echo "complete: $(grep -l '^DONE' $D/log.g* 2>/dev/null | wc -l) done in $D"
