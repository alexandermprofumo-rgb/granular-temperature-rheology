#!/bin/bash
# Is the production n a STEADY-STATE quantity, or a transient?
#
# Every sweep in this project shears at gdot=1e-3 for 40k+80k steps at
# dt=1e-3, i.e. total strain 0.12. Dense granular fabric needs strain of order
# 1 to reach steady state, and the logs show it: mu drifts +12% to +30% WITHIN
# the measurement window in every sweep, including the published Pconf=2 one
# (+19.3%).
#
# n is a slope, so a Theta-independent drift cancels; refitting early vs late
# halves shifts n by only 0.01-0.05 (1-2 sigma). So the published numbers are
# not obviously wrong. But the shifts at Pconf=10 are systematically NEGATIVE,
# implying n keeps falling as strain accumulates -- and the gdot sweep agrees:
# at the two genuinely steady rates (drift < 2%), mu_g=0.3 gives n = 0.100 and
# 0.096, well below the production 0.144.
#
# Two readings, and they are not distinguishable from existing data:
#   (i)  n is I-dependent, and the low-I production point genuinely differs;
#   (ii) n is I-INdependent near 0.10, and the production 0.144 is a transient
#        artifact of insufficient strain.
#
# This run decides it: same gdot=1e-3 and same I as production, but sheared to
# strain 1.5 (1.5e6 steps) with the measurement taken over the last strain 1.0.
# If n converges to ~0.10 the production numbers are transients; if it stays at
# ~0.144 then n really does depend on I.
#
# Usage:  ./run_steadystate.sh [njobs]     (default 12)
set -u
cd "$(dirname "$0")"

NJOBS=${1:-12}
D=sweep_steadystate
MUG=(0.1 0.3)
TG=(0.001 0.003 0.008 0.02)
SEEDS=(1 2 3)
N=4000; PHI=0.80; PCONF=10.0; GDOT=1.0e-3
NEQUIL=500000     # strain 0.5 of equilibration
NMEAS=1000000     # measurement over strain 1.0

if command -v gtimeout >/dev/null 2>&1; then TO="gtimeout 10800"; else TO="timeout 10800"; fi
mkdir -p $D
for s in "${SEEDS[@]}"; do
  [ -f "$D/data.ss_seed${s}" ] || python3 gen_data.py --N $N --phi $PHI --seed $s --out "$D/data.ss_seed${s}"
done

one(){
  local mg=$1 T=$2 s=$3 label="mu${mg}_T${T}_s${s}"
  if [ -f "$D/log.${label}" ] && grep -q "^DONE" "$D/log.${label}"; then return; fi
  ( cd "$D" && $TO lmp_serial -in ../in.granular_2d \
      -var datafile "data.ss_seed${s}" -var mu_g $mg -var gdot $GDOT \
      -var Pconf $PCONF -var Tgran $T -var seed $s -var label "$label" \
      -var nequil $NEQUIL -var nmeas $NMEAS > "run.${label}.out" 2>&1 )
}

for mg in "${MUG[@]}"; do for T in "${TG[@]}"; do for s in "${SEEDS[@]}"; do
  one $mg $T $s &
  while [ "$(jobs -rp | wc -l)" -ge "$NJOBS" ]; do wait -n 2>/dev/null || sleep 1; done
done; done; done
wait
echo "STEADYSTATE_COMPLETE"
