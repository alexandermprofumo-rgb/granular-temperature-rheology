#!/bin/bash
# ATHERMAL CONTROL, ROUND 2.  Round 1 (sweep_nothermo2) returned a split
# verdict that cannot be interpreted as it stands:
#
#   mu_g = 0.3   athermal mu agrees with the thermostatted curve at the same
#                Theta to 0.6% (0.4 sigma)  -> the bath is validated there
#   mu_g = 0.15  athermal mu is 15-20% HIGHER at matched Theta, I, e, mu_g and
#                packing (21-24 sigma)      -> the bath is NOT validated there
#   mu_g = 0     untestable: with no bath the frictionless pack unjams to a
#                gas (Z ~ 0.9-1.1, Theta ~ 1e-1) and is not a dense flow at all
#
# Two things are needed before that can go in a paper.
#
# 1. WHERE does the discrepancy live? Round 1 sampled two frictions and they
#    disagreed with each other. Fill in mu_g = 0.05, 0.1, 0.2, 0.5 at e = 0.5.
# 2. Is the round-1 result initial-condition independent? The kick control was
#    run only at mu_g = 0 -- the cell that turned out to be a gas -- so it
#    tested the one state that does not matter and found (unsurprisingly) that
#    the gas remembers its kick. Repeat it at mu_g = 0.15 and 0.3.
#
# Usage:  ./run_nothermo2b.sh [njobs]
set -u
cd "$(dirname "$0")"
NJOBS=${1:-14}
while ! pmset -g ps 2>/dev/null | grep -q "AC Power"; do
  echo "$(date '+%H:%M:%S')  on battery -- waiting for AC"; sleep 60
done
caffeinate -dimsu -w $$ &
disown $!   # BUG FIXED 2026-08-28: without this, bash's bare `wait` below waits on
            # every backgrounded job the script started, INCLUDING this caffeinate --
            # which only exits when the script does. Deadlock: all lmp_serial
            # jobs finished, the driver sat alive with no children for 5+ hours.
            # disown removes it from the job table wait watches, without touching
            # -w $$, so it still releases the wake assertion when the script exits.
echo "$(date '+%H:%M:%S')  on AC -- starting"
D=sweep_nothermo2; mkdir -p $D
throttle(){ while [ "$(jobs -rp | wc -l)" -ge "$NJOBS" ]; do wait -n 2>/dev/null || sleep 1; done; }
launch(){ local lab=$1; shift
  if [ -f "$D/log.${lab}" ] && grep -q "^DONE" "$D/log.${lab}"; then return; fi
  rm -f "$D/log.${lab}"
  ( cd "$D" && lmp_serial "$@" -var label "$lab" </dev/null > "run.${lab}.out" 2>&1 ) &
  throttle; }
# 1. fill in the friction axis
for mg in 0.05 0.1 0.2 0.5; do for s in 1 2; do
  launch "nt_g1.0e-3_e0.5_mu${mg}_s${s}" -in ../in.granular_2d_nothermo_ss \
    -var datafile "data.re_seed${s}" -var mu_g $mg -var gdot 1.0e-3 \
    -var Pconf 10.0 -var e_rest 0.5 -var seed $s \
    -var nequil 1000000 -var nmeas 1000000
done; done
# 2. kick control where it matters
for mg in 0.15 0.3; do for s in 1 2; do
  launch "ntk_g1.0e-3_e0.5_mu${mg}_s${s}" -in ../in.granular_2d_nothermo_ss \
    -var datafile "data.re_seed${s}" -var mu_g $mg -var gdot 1.0e-3 \
    -var Pconf 10.0 -var e_rest 0.5 -var seed $s -var t_kick 1.0e-3 \
    -var nequil 1000000 -var nmeas 1000000
done; done
wait
echo "NOTHERMO2B_COMPLETE"
