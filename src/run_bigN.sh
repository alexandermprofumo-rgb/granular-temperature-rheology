#!/bin/bash
# LARGE-N FRICTIONLESS -- the one necessary follow-up.
#
# The pre-registered finite-size test FAILED. The mu_g = 1 failure is a jamming
# artefact (5 setpoints, near-isostatic) and more N will not fix it. But the
# FRICTIONLESS row -- the paper's central number -- rises monotonically
# 0.019 -> 0.030 -> 0.039 -> 0.042 with increments +0.010, +0.009, +0.003.
# That looks like convergence to n ~ 0.043, but "looks like" is not a measured
# limit, and this is the value constraint 2 rests on.
#
# N = 16000 and 32000 at mu_g = 0 only. Cost scales with N, so 32000 dominates.
# ~40 runs, ~7 h at 12 jobs.
set -u
cd "$(dirname "$0")"
NJOBS=${1:-12}
D=sweep_size2d
# Python interpreter.  Override with  PY=/path/to/python  in the environment;
# otherwise prefer the project virtualenv, then fall back to python3 on PATH.
if [ -z "${PY:-}" ]; then
  _venv="$(cd "$(dirname "$0")/.." 2>/dev/null && pwd)/.venv/bin/python"
  if [ -x "$_venv" ]; then PY="$_venv"; else PY=python3; fi
fi
TG=(0.0005 0.001 0.0015 0.003 0.005 0.008 0.013 0.02 0.03)
for N in 16000 32000; do for s in 1 2; do
  f="$D/data.N${N}_seed${s}"
  [ -f "$f" ] || $PY gen_data.py --N $N --phi 0.80 --seed $s --out "$f"
done; done
throttle(){ while [ "$(jobs -rp | wc -l)" -ge "$NJOBS" ]; do wait -n 2>/dev/null || sleep 1; done; }
launch(){ local lab=$1; shift
  if [ -f "$D/log.${lab}" ] && grep -q "^DONE" "$D/log.${lab}"; then return; fi
  rm -f "$D/log.${lab}"
  ( cd "$D" && nice -n 5 lmp_serial "$@" -var label "$lab" > "run.${lab}.out" 2>&1 ) &
  throttle; }
for N in 16000 32000; do for T in "${TG[@]}"; do for s in 1 2; do
  launch "N${N}_mu0.0_T${T}_s${s}" -in ../in.granular_2d_ss \
    -var datafile "data.N${N}_seed${s}" -var mu_g 0.0 -var gdot 1.0e-3 \
    -var Pconf 10.0 -var Tgran $T -var seed $s \
    -var nequil 500000 -var nmeas 500000 -var cdump_every 25000
done; done; done
wait
echo "BIGN_COMPLETE"
