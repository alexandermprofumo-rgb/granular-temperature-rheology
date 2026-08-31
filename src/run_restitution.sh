#!/bin/bash
# RESTITUTION SCAN -- e = 0.5 everywhere in the project, never varied.
#
# A robustness check rather than expected new physics: "you tested one
# dissipation value" is otherwise an objection with no answer. e = 0.5 is
# re-run inside this sweep rather than borrowed from sweep_steady2d, so the
# reference passes through the same deck as the test points, the same
# same-pipeline discipline applied elsewhere to chi.
#
# 3 restitutions x 3 frictions x 9 Theta x 2 seeds = 162 runs, ~5 h at 12 jobs.
# in.granular_2d_sens_ss now also reports the rotational temperature.
set -u
cd "$(dirname "$0")"
NJOBS=${1:-12}
D=sweep_restit; mkdir -p $D
NEQ=500000; NME=500000; CDUMP=25000
for s in 1 2; do
  [ -f "$D/data.re_seed${s}" ] || cp sweep_matched2d/data.m2d_seed${s} "$D/data.re_seed${s}"
done
TG=(0.0005 0.001 0.0015 0.003 0.005 0.008 0.013 0.02 0.03)

throttle(){ while [ "$(jobs -rp | wc -l)" -ge "$NJOBS" ]; do wait -n 2>/dev/null || sleep 1; done; }
launch(){ local lab=$1; shift
  if [ -f "$D/log.${lab}" ] && grep -q "^DONE" "$D/log.${lab}"; then return; fi
  rm -f "$D/log.${lab}"
  ( cd "$D" && nice -n 5 lmp_serial "$@" -var label "$lab" > "run.${lab}.out" 2>&1 ) &
  throttle; }

for e in 0.1 0.5 0.9; do for mg in 0.0 0.15 1.0; do
 for T in "${TG[@]}"; do for s in 1 2; do
  launch "e${e}_mu${mg}_T${T}_s${s}" -in ../in.granular_2d_sens_ss \
    -var datafile "data.re_seed${s}" -var mu_g $mg -var gdot 1.0e-3 \
    -var Pconf 10.0 -var Tgran $T -var seed $s -var e_rest $e -var kt_spec NULL \
    -var nequil $NEQ -var nmeas $NME -var cdump_every $CDUMP
done; done; done; done
wait
echo "RESTITUTION_COMPLETE"
