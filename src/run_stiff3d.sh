#!/bin/bash
# 3D stiffness lever: 3 E (100x) x 3 frictions x 12 Theta x 2 seeds = 216 runs.
# Bidimensionalises constraint 8's empirical half. ~11.5 h at 12 jobs.
set -u
cd "$(dirname "$0")"
NJOBS=${1:-12}
D=sweep_lev_E3d; mkdir -p $D
# Python interpreter.  Override with  PY=/path/to/python  in the environment;
# otherwise prefer the project virtualenv, then fall back to python3 on PATH.
if [ -z "${PY:-}" ]; then
  _venv="$(cd "$(dirname "$0")/.." 2>/dev/null && pwd)/.venv/bin/python"
  if [ -x "$_venv" ]; then PY="$_venv"; else PY=python3; fi
fi
for s in 1 2; do
  [ -f "$D/data.g3d_seed${s}" ] || cp sweep_run_3d/data.granular3d_seed${s} "$D/data.g3d_seed${s}"
done
TG=(0.0005 0.001 0.0015 0.003 0.005 0.008 0.013 0.02 0.03 0.05 0.08 0.12)
throttle(){ while [ "$(jobs -rp | wc -l)" -ge "$NJOBS" ]; do wait -n 2>/dev/null || sleep 1; done; }
launch(){ local lab=$1; shift
  if [ -f "$D/log.${lab}" ] && grep -q "^DONE" "$D/log.${lab}"; then return; fi
  rm -f "$D/log.${lab}"
  ( cd "$D" && nice -n 5 lmp_serial "$@" -var label "$lab" > "run.${lab}.out" 2>&1 ) &
  throttle; }
for E in 1.0e4 1.0e5 1.0e6; do
  dt=$($PY -c "print(f'{0.001*($E/1.0e5)**-0.4:.8f}')")
  NS=$($PY -c "print(int(round(0.5/(1.0e-3*$dt))))")
  for mg in 0.1 0.3 1.0; do for T in "${TG[@]}"; do for s in 1 2; do
    launch "E${E}_mu${mg}_T${T}_s${s}" -in ../in.granular_3d_stiff_ss \
      -var datafile "data.g3d_seed${s}" -var mu_g $mg -var gdot 1.0e-3 \
      -var Pconf 10.0 -var Tgran $T -var seed $s -var Emod $E -var dt $dt \
      -var nequil $NS -var nmeas $NS -var cdump_every 50000
done; done; done; done
wait
echo "STIFF3D_COMPLETE"
