#!/bin/bash
# Two independent probes, run together. Both attack assumptions rather than
# mechanisms, which is why they rank above another mechanism hunt.
#
# (A) nothermo : IS THE MEASUREMENT PROTOCOL VALID?
#     Every n in this project came from a Langevin thermostat imposing Theta,
#     and Tier 0-A showed n depends on the damping time. Here the bath is
#     removed and restitution scans Theta instead (less dissipation -> hotter)
#     at fixed I. Compare the resulting (Theta, mu) points against the
#     thermostatted sweep_matched2d at the same mu_g / Pconf / gdot. One curve
#     => protocol validated. Two curves => the central quantity is protocol
#     dependent. Smoke test: e=0.9, mu_g=0.3 gives Theta=3.0e-4, comfortably
#     inside the thermostatted range 5.4e-5..1.1e-3, so the curves overlap.
#     Needs ~3x the run length: without a bath the heating/dissipation balance
#     and the fabric must self-organise (measurement strain 0.25 vs 0.12).
#
# (B) nofI : IS n EVEN INDEPENDENT OF I?
#     mu*Theta^n = F(I) asserts that n is a constant and ALL the I dependence
#     sits in F. Nobody has tested it -- every run in the campaign sits at
#     I = 3.2e-4. Scanning gdot at fixed Pconf scans I directly. If n drifts
#     with I the assumed separable form is wrong and the honest statement is
#     mu = F(I, Theta) with no power law. Expect the top of the gdot range to
#     break: shear heating grows as gdot^2 and will overwhelm the thermostat,
#     the same failure that killed the pressure lever. That boundary is itself
#     worth documenting, so the analysis gates on I drift and Theta control
#     rather than assuming every cell is usable.
#
# Usage:  ./run_protocol.sh <which: nothermo|nofI> [njobs]
set -u
cd "$(dirname "$0")"

WHICH=${1:?nothermo or nofI}
NJOBS=${2:-3}
SEEDS=(1 2 3)
N=4000; PHI=0.80; PCONF=10.0

if command -v gtimeout >/dev/null 2>&1; then TO="gtimeout 7200"; else TO="timeout 7200"; fi

case $WHICH in
nothermo)
  D=sweep_nothermo
  EREST=(0.3 0.5 0.7 0.8 0.9 0.95)
  MUG=(0.1 0.3)
  GDOT=1.0e-3
  mkdir -p $D
  for s in "${SEEDS[@]}"; do
    [ -f "$D/data.nt_seed${s}" ] || python3 gen_data.py --N $N --phi $PHI --seed $s --out "$D/data.nt_seed${s}"
  done
  for e in "${EREST[@]}"; do for mg in "${MUG[@]}"; do for s in "${SEEDS[@]}"; do
    label="e${e}_mu${mg}_s${s}"
    if [ -f "$D/log.${label}" ] && grep -q "^DONE" "$D/log.${label}"; then continue; fi
    ( cd "$D" && $TO lmp_serial -in ../in.granular_2d_nothermo \
        -var datafile "data.nt_seed${s}" -var mu_g $mg -var e_rest $e \
        -var gdot $GDOT -var Pconf $PCONF -var seed $s -var label "$label" \
        -var nequil 150000 -var nmeas 250000 > "run.${label}.out" 2>&1 ) &
    while [ "$(jobs -rp | wc -l)" -ge "$NJOBS" ]; do wait -n 2>/dev/null || sleep 1; done
  done; done; done
  wait; echo "NOTHERMO_COMPLETE" ;;

nofI)
  D=sweep_nofI
  GDOTS=(3.0e-4 1.0e-3 3.0e-3 1.0e-2)
  MUG=(0.1 0.3)
  TG=(0.001 0.003 0.008 0.02)
  mkdir -p $D
  for s in "${SEEDS[@]}"; do
    [ -f "$D/data.nofI_seed${s}" ] || python3 gen_data.py --N $N --phi $PHI --seed $s --out "$D/data.nofI_seed${s}"
  done
  for g in "${GDOTS[@]}"; do for mg in "${MUG[@]}"; do for T in "${TG[@]}"; do for s in "${SEEDS[@]}"; do
    label="g${g}_mu${mg}_T${T}_s${s}"
    if [ -f "$D/log.${label}" ] && grep -q "^DONE" "$D/log.${label}"; then continue; fi
    ( cd "$D" && $TO lmp_serial -in ../in.granular_2d \
        -var datafile "data.nofI_seed${s}" -var mu_g $mg -var gdot $g \
        -var Pconf $PCONF -var Tgran $T -var seed $s -var label "$label" \
        -var nequil 40000 -var nmeas 80000 > "run.${label}.out" 2>&1 ) &
    while [ "$(jobs -rp | wc -l)" -ge "$NJOBS" ]; do wait -n 2>/dev/null || sleep 1; done
  done; done; done; done
  wait; echo "NOFI_COMPLETE" ;;
*) echo "unknown: $WHICH"; exit 1 ;;
esac
