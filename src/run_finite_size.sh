#!/bin/bash
# FINITE-SIZE SCALING AT P = 10 -- never done, in either dimension.
#
# The project has only ever run N = 4000 at the production state (plus N = 2000
# at Pconf = 2, which audit_pi_groups flags as three Pi-groups away and
# therefore uninterpretable). "Does N = 4000 suffice?" is the most obvious
# referee question with no answer at all.
#
# N = 1000, 2000, 8000 against the existing 4000, at three frictions spanning
# the curve (0 = the frictionless anchor, 0.15 = the peak, 1.0 = the collapsed
# high-friction branch), strain 1.0, jamming-gated like everything else.
#
# Cost scales ~linearly in N, so N = 8000 dominates: ~4.6 h at 12 jobs.
set -u
cd "$(dirname "$0")"
NJOBS=${1:-12}
D=sweep_size2d; mkdir -p $D
NEQ=500000; NME=500000; CDUMP=25000
# Python interpreter.  Override with  PY=/path/to/python  in the environment;
# otherwise prefer the project virtualenv, then fall back to python3 on PATH.
if [ -z "${PY:-}" ]; then
  _venv="$(cd "$(dirname "$0")/.." 2>/dev/null && pwd)/.venv/bin/python"
  if [ -x "$_venv" ]; then PY="$_venv"; else PY=python3; fi
fi
TG=(0.0005 0.001 0.0015 0.003 0.005 0.008 0.013 0.02 0.03)

for N in 1000 2000 8000; do for s in 1 2; do
  f="$D/data.N${N}_seed${s}"
  [ -f "$f" ] || $PY gen_data.py --N $N --phi 0.80 --seed $s --out "$f"
done; done

throttle(){ while [ "$(jobs -rp | wc -l)" -ge "$NJOBS" ]; do wait -n 2>/dev/null || sleep 1; done; }
launch(){ local lab=$1; shift
  if [ -f "$D/log.${lab}" ] && grep -q "^DONE" "$D/log.${lab}"; then return; fi
  rm -f "$D/log.${lab}"
  ( cd "$D" && nice -n 5 lmp_serial "$@" -var label "$lab" > "run.${lab}.out" 2>&1 ) &
  throttle; }

for N in 1000 2000 8000; do for mg in 0.0 0.15 1.0; do
 for T in "${TG[@]}"; do for s in 1 2; do
  launch "N${N}_mu${mg}_T${T}_s${s}" -in ../in.granular_2d_ss \
    -var datafile "data.N${N}_seed${s}" -var mu_g $mg -var gdot 1.0e-3 \
    -var Pconf 10.0 -var Tgran $T -var seed $s \
    -var nequil $NEQ -var nmeas $NME -var cdump_every $CDUMP
done; done; done; done
wait
echo "FINITE_SIZE_COMPLETE"
