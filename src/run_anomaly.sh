#!/bin/bash
# The mu_g ~ 0.3 anomaly: dense friction grid at four pressures.
#
# WHAT THE ANOMALY IS. kappa = E/P organises n at mu_g = 0.1 and 1.0 (pressure
# and stiffness levers agree to 0.8 and 0.3 sigma) but fails at mu_g = 0.3,
# where the two levers disagree 7x (b = +0.0479 +/- 0.0081 vs +0.0066 +/-
# 0.0023, 4.9 sigma). Both levers independently localise their trouble to the
# same friction, just past the peak of n(mu_g).
#
# THE DISCRIMINATING QUESTION. Two readings fit the facts so far:
#   (b) PEAK PROXIMITY -- n is locally flat at its maximum, so anything that
#       shifts the peak produces a large apparent dn there. Then the spike in
#       pressure-sensitivity should TRACK the peak as pressure moves it.
#   (a) REAL CROSSOVER -- something changes character near mu_g ~ 0.3
#       (e.g. sticking- to sliding-dominated dissipation). Then the spike
#       stays PINNED near 0.3 even as the peak moves.
# A dense n(mu_g, P) grid separates these directly: measure both b(mu_g) and
# the peak location mu_g*(P) from the same runs.
#
# This also delivers the peak-tracking pathway for free -- mu_g*(P) is read
# off the same grid rather than needing its own sweep.
#
# Same Pi-group scalings as run_piscan.sh (gdot ~ sqrt(P), tdamp ~ 1/sqrt(P),
# Tgran ~ P, dt ~ 1/sqrt(P)) so only kappa moves with pressure. P=2 is
# excluded: it sits at the rigid-grain jamming point and its cold setpoints
# are physically inaccessible (P = 4.7 +/- 8.3, Z = 1.15).
#
# mu_g = 0.1, 0.3, 1.0 already exist in sweep_piscan and are skipped.
#
# RUN IN TWO STAGES. Stage 1 used MUG=(0.15 0.2 0.25 0.35 0.4 0.5) on the
# assumption -- taken from sweep_matched2d -- that n(mu_g) peaks near 0.15.
# It does not under THIS protocol: the piscan runs use a pressure-scaled Theta
# window and drop the I-deviant coldest setpoint, and with those n is already
# DECREASING at mu_g = 0.1 for P >= 10. The peak-locating parabola therefore
# pinned mu_g* at the grid edge, and the apparent "peak shift 0.28 -> 0.10"
# was an edge artifact, not a measurement.
#
# Stage 2 (MUG=(0.03 0.05 0.07 0.5)) extends BELOW 0.1 so the peak is
# bracketed at every pressure, which is what the peak-proximity vs crossover
# discriminator actually requires. Stage 1's b(mu_g) result is unaffected --
# it needs no peak location, only n at fixed mu_g across P -- and already
# shows a clean interior maximum at mu_g = 0.3, ~4-8x the wings.
#
# Usage:  ./run_anomaly.sh [njobs]     (default 14)
set -u
cd "$(dirname "$0")"

NJOBS=${1:-14}
D=sweep_piscan
PCONF=(5.0 10.0 25.0 50.0)
MUG=(0.03 0.05 0.07 0.5)
TG10=(0.001 0.003 0.008 0.02)
SEEDS=(1 2 3)
N=4000; PHI=0.80

if command -v gtimeout >/dev/null 2>&1; then TO="gtimeout 7200"; else TO="timeout 7200"; fi
mkdir -p $D
for s in "${SEEDS[@]}"; do
  [ -f "$D/data.pi_seed${s}" ] || python3 gen_data.py --N $N --phi $PHI --seed $s --out "$D/data.pi_seed${s}"
done

one(){
  local P=$1 mg=$2 T10=$3 s=$4
  local label="P${P}_mu${mg}_T${T10}_s${s}"
  if [ -f "$D/log.${label}" ] && grep -q "^DONE" "$D/log.${label}"; then return; fi
  read gdot tdamp dt Tg < <(python3 -c "
import math
P=$P; T10=$T10
print(f'{3.162e-4*math.sqrt(P):.6e} {1.581/math.sqrt(P):.6f} {0.001*math.sqrt(10.0/P):.6e} {T10*P/10.0:.6e}')")
  ( cd "$D" && $TO lmp_serial -in ../in.granular_2d_piscan \
      -var datafile "data.pi_seed${s}" -var mu_g $mg -var Pconf $P \
      -var gdot $gdot -var tdamp $tdamp -var dt $dt -var Tgran $Tg \
      -var seed $s -var label "$label" -var nequil 40000 -var nmeas 80000 \
      > "run.${label}.out" 2>&1 )
}

for mg in "${MUG[@]}"; do for P in "${PCONF[@]}"; do for T in "${TG10[@]}"; do for s in "${SEEDS[@]}"; do
  one $P $mg $T $s &
  while [ "$(jobs -rp | wc -l)" -ge "$NJOBS" ]; do wait -n 2>/dev/null || sleep 1; done
done; done; done; done
wait
echo "ANOMALY_COMPLETE"
