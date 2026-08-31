#!/bin/bash
# Tighten the pressure lever at mu_g = 0.1 and 1.0.
#
# The kappa two-lever result rests on ONE friction value. At mu_g=0.3 the
# pressure and stiffness levers disagree at 3.9 sigma (b = +0.0556 +/- 0.0125
# vs +0.0066 +/- 0.0023). At mu_g=0.1 and 1.0 they agree -- but with pressure-
# lever errors of +/-0.041 and +/-0.019, which would agree with almost
# anything. Those two points are uninformative, not confirming, and they are
# the weakest link in the strongest new result.
#
# Fix: 7 temperature setpoints instead of 4 (more lever arm in ln Theta for
# each n fit) and 5 seeds instead of 3. Existing runs are skipped, so this
# only adds what is missing.
#
# P=2 is NOT included: its cold setpoints are physically inaccessible, not
# noisy. At P=2 with E=1e5 the overlap is (P/E)^(2/3) ~ 7e-4, so the packing
# sits essentially at the rigid-grain jamming point -- a cold system loses
# contacts, pressure collapses, and the barostat hunts (P = 4.67 +/- 8.30 at
# the coldest setpoint, Z = 1.15). That is a real ceiling on kappa at fixed I
# and Pi_damp, not something more sampling would cure.
#
# Same scalings as run_piscan.sh so only kappa moves.
#
# Usage:  ./run_piscan_tighten.sh [njobs]     (default 14)
set -u
cd "$(dirname "$0")"

NJOBS=${1:-14}
D=sweep_piscan
PCONF=(5.0 10.0 25.0 50.0)
MUG=(0.1 1.0)
TG10=(0.001 0.0018 0.003 0.005 0.008 0.013 0.02)
SEEDS=(1 2 3 4 5)
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

for P in "${PCONF[@]}"; do for mg in "${MUG[@]}"; do for T in "${TG10[@]}"; do for s in "${SEEDS[@]}"; do
  one $P $mg $T $s &
  while [ "$(jobs -rp | wc -l)" -ge "$NJOBS" ]; do wait -n 2>/dev/null || sleep 1; done
done; done; done; done
wait
echo "PISCAN_TIGHTEN_COMPLETE"
