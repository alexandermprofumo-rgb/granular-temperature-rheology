#!/bin/bash
# The corrected pressure lever: scan kappa = E/P with EVERY other
# dimensionless group held fixed.
#
# kappa is the one group with a plausible claim to being the missing variable
# in n, and it has never been moved without confounds. The stiffness sweep
# moved it at fixed pressure (b = +0.0066 +/- 0.0023); this moves it by
# pressure instead, over 25x rather than 100x but along an orthogonal
# direction in (E,P) space. Agreement between the two b values would confirm
# kappa; disagreement would prove E and P do NOT enter only as E/P.
#
# Four things scale with P so that ONLY kappa moves (see audit_pi_groups.py):
#   gdot   ~ sqrt(P)    holds I fixed at 3.162e-4
#   tdamp  ~ 1/sqrt(P)  holds Pi_damp fixed at 1.581  <-- the missing piece
#   Tgran  ~ P          holds the dimensionless Theta window fixed
#   dt     ~ 1/sqrt(P)  stability, and dt*gdot invariant so a fixed step
#                       count covers a fixed strain
#
# Pi_damp is why the original lever (Result 6) failed. t_damp was hardcoded at
# 0.5 in absolute units, so raising P raised the DIMENSIONLESS damping time --
# a bath that damps slowly compared to grain dynamics -- exactly where shear
# heating was worst. Smoke test at Pconf=50: the old lever gave Theta/Tgran =
# 6.21 (thermostat lost control, fitted n became nonsense); with tdamp scaled
# it gives 0.093, inside the normal production range.
#
# HOLD OUT P=50 when fitting. Fit b on P = 2..25, predict P=50, then look.
# Every collapse in this project that was fitted on all the data survived and
# then died to an orthogonal test; the discipline is to predict first.
#
# Usage:  ./run_piscan.sh [njobs]     (default 14)
set -u
cd "$(dirname "$0")"

NJOBS=${1:-14}
D=sweep_piscan
PCONF=(2.0 5.0 10.0 25.0 50.0)
MUG=(0.1 0.3 1.0)
TG10=(0.001 0.003 0.008 0.02)     # the Pconf=10 grid; scaled by P/10 below
SEEDS=(1 2 3)
N=4000; PHI=0.80

if command -v gtimeout >/dev/null 2>&1; then TO="gtimeout 7200"; else TO="timeout 7200"; fi
mkdir -p $D
for s in "${SEEDS[@]}"; do
  [ -f "$D/data.pi_seed${s}" ] || python3 gen_data.py --N $N --phi $PHI --seed $s --out "$D/data.pi_seed${s}"
done

one(){
  local P=$1 mg=$2 T10=$3 s=$4
  read gdot tdamp dt Tg < <(python3 -c "
import math
P=$P; T10=$T10
print(f'{3.162e-4*math.sqrt(P):.6e} {1.581/math.sqrt(P):.6f} {0.001*math.sqrt(10.0/P):.6e} {T10*P/10.0:.6e}')")
  local label="P${P}_mu${mg}_T${T10}_s${s}"
  if [ -f "$D/log.${label}" ] && grep -q "^DONE" "$D/log.${label}"; then return; fi
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
echo "PISCAN_COMPLETE"
