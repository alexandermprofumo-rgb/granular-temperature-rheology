#!/bin/bash
# Four independent post-hoc sweeps. $1 = seed, $2 = which.
#   peak  : extend Tier 0-A to the FULL mu_g range at 3 tdamp values. Tier 0-A
#           only sampled mu_g={0,0.1,0.3,1.0}, where the small-tdamp curves
#           were still RISING at the edge -- so we do not actually know
#           whether a peak exists away from the production tdamp=0.5. The
#           whole paper rests on the peak being real. Reuses sweep_tier0A so
#           already-completed runs are skipped.
#   frz   : frozen-rotation runs (campaign item 9), the last untested framework.
#   sens  : restitution and tangential-stiffness sensitivity (item 10). The
#           report calls this "a question any careful referee will ask", and it
#           discriminates: generalised isostaticity predicts sensitivity to
#           mu_g but NOT to kt/kn; tangential-spring pictures predict the reverse.
#   size  : finite-size check at the peak (item 3 / S2). N=4000 already exists
#           from production, so only 1000/2000/8000 are run.
set -e
SEED=$1; WHICH=$2
if command -v gtimeout >/dev/null 2>&1; then TO="gtimeout 3600"; else TO="timeout 3600"; fi
TG=(0.0005 0.002 0.006 0.015); PHI=0.80; PCONF=2.0; GDOT=1.0e-3

run(){ # $1=dir $2=label $3...=extra lmp args
  local dir=$1 label=$2; shift 2
  [ -f "$dir/log.${label}" ] && grep -q "^DONE" "$dir/log.${label}" && { echo "skip ${label}"; return; }
  echo "=== ${label} ==="; set +e
  ( cd "$dir" && $TO lmp_serial "$@" -var label "$label" > "run.${label}.out" 2>&1 ); set -e
}

case $WHICH in
peak)
  D=sweep_tier0A; mkdir -p $D
  DF="data.tier0A_seed${SEED}"
  [ -f "$D/$DF" ] || python3 gen_data.py --N 4000 --phi $PHI --seed $SEED --out "$D/$DF"
  for mg in 0.001 0.01 0.03 2.0 5.0; do for td in 0.15 0.5 1.5; do for T in "${TG[@]}"; do
    run $D "A_mu${mg}_T${T}_td${td}_s${SEED}" -in ../in.tier0_thermo_2d -var datafile "$DF" \
        -var mu_g $mg -var gdot $GDOT -var Pconf $PCONF -var Tgran $T -var seed $SEED \
        -var tdamp $td -var nequil 40000 -var nprod 80000
  done; done; done ;;
frz)
  D=sweep_frozenrot; mkdir -p $D
  DF="data.frz_seed${SEED}"
  [ -f "$D/$DF" ] || python3 gen_data.py --N 4000 --phi $PHI --seed $SEED --out "$D/$DF"
  for mg in 0.0 0.001 0.01 0.03 0.1 0.3 1.0 2.0 5.0; do for T in "${TG[@]}"; do
    run $D "mu${mg}_T${T}_s${SEED}" -in ../in.granular_2d_frozenrot -var datafile "$DF" \
        -var mu_g $mg -var gdot $GDOT -var Pconf $PCONF -var Tgran $T -var seed $SEED \
        -var nequil 40000 -var nmeas 80000
  done; done ;;
sens)
  D=sweep_sens; mkdir -p $D
  DF="data.sens_seed${SEED}"
  [ -f "$D/$DF" ] || python3 gen_data.py --N 4000 --phi $PHI --seed $SEED --out "$D/$DF"
  for mg in 0.0 0.3 1.0; do
    for e in 0.2 0.5 0.9; do for T in "${TG[@]}"; do
      run $D "e${e}_kNULL_mu${mg}_T${T}_s${SEED}" -in ../in.granular_2d_sens -var datafile "$DF" \
          -var mu_g $mg -var gdot $GDOT -var Pconf $PCONF -var Tgran $T -var seed $SEED \
          -var e_rest $e -var kt_spec NULL -var nequil 40000 -var nmeas 80000
    done; done
    for kt in 1.0e4 1.0e5; do for T in "${TG[@]}"; do
      run $D "e0.5_k${kt}_mu${mg}_T${T}_s${SEED}" -in ../in.granular_2d_sens -var datafile "$DF" \
          -var mu_g $mg -var gdot $GDOT -var Pconf $PCONF -var Tgran $T -var seed $SEED \
          -var e_rest 0.5 -var kt_spec $kt -var nequil 40000 -var nmeas 80000
    done; done
  done ;;
size)
  D=sweep_size; mkdir -p $D
  for N in 1000 2000 8000; do
    DF="data.N${N}_seed${SEED}"
    [ -f "$D/$DF" ] || python3 gen_data.py --N $N --phi $PHI --seed $SEED --out "$D/$DF"
    for mg in 0.0 0.3 1.0; do for T in "${TG[@]}"; do
      run $D "N${N}_mu${mg}_T${T}_s${SEED}" -in ../in.granular_2d -var datafile "$DF" \
          -var mu_g $mg -var gdot $GDOT -var Pconf $PCONF -var Tgran $T -var seed $SEED \
          -var nequil 40000 -var nmeas 80000
    done; done
  done ;;
esac
echo "SEED ${SEED} ${WHICH} COMPLETE"
