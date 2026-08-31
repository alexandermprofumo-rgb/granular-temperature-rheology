#!/bin/bash
# chi (and hence dZ_eff) at varying grain stiffness -- the decisive test of
# the generalised-isostaticity collapse.
#
# The collapse currently sits at 1.5 sigma, but every dZ_eff value ever
# measured was reached by turning ONE knob, mu_g. So "n depends on dZ_eff" has
# never been separated from "n and dZ_eff both depend on mu_g".
#
# The stiffness sweep supplies an orthogonal knob, and it already looks
# damaging. At mu_g=0.3, stiffening the grains drops Z from 4.01 to 3.32 while
# n RISES 0.107 -> 0.177; along the friction axis at fixed E, the same drop in
# Z comes with n FALLING 0.165 -> 0.075. Opposite responses at matched Z, so n
# is not a function of coordination alone.
#
# That argument used Z, not dZ_eff. It holds only if chi is roughly
# independent of Emod at fixed mu_g -- if chi instead moves with stiffness,
# Z_c(chi) could absorb the difference and the collapse would survive. This
# sweep measures chi directly and settles it.
#
# Timestep and step counts scale as in run_stiffness.sh so every modulus
# covers the same physical duration.
#
# Usage:  ./run_tracking_stiff.sh [njobs]     (default 10)
set -u
cd "$(dirname "$0")"

NJOBS=${1:-10}
D=sweep_tracking_stiff
EMOD=(1.0e4 1.0e5 1.0e6)
MUG=(0.1 0.3 1.0)
TG=(0.001 0.003 0.008)
SEEDS=(1 2 3)
N=4000; PHI=0.80; PCONF=10.0; GDOT=1.0e-3

if command -v gtimeout >/dev/null 2>&1; then TO="gtimeout 5400"; else TO="timeout 5400"; fi
mkdir -p $D
for s in "${SEEDS[@]}"; do
  [ -f "$D/data.trks_seed${s}" ] || python3 gen_data.py --N $N --phi $PHI --seed $s --out "$D/data.trks_seed${s}"
done

one(){
  local E=$1 mg=$2 T=$3 s=$4
  local label="E${E}_mu${mg}_T${T}_s${s}"
  if [ -f "$D/log.${label}" ] && grep -q "^DONE" "$D/log.${label}"; then return; fi
  read dt nr ne nm < <(python3 in.granular_2d_stiff.py "$E")
  ( cd "$D" && $TO lmp_serial -in ../in.contact_tracking_forces_2d \
      -var datafile "data.trks_seed${s}" -var mu_g $mg -var gdot $GDOT \
      -var Pconf $PCONF -var Tgran $T -var seed $s -var label "$label" \
      -var Emod $E -var dt $dt -var nrelax $nr -var nequil $ne \
      -var ntrack 3000 -var dump_every 15 -var cdump_every 30 \
      > "run.${label}.out" 2>&1 )
}

for E in "${EMOD[@]}"; do for mg in "${MUG[@]}"; do for T in "${TG[@]}"; do for s in "${SEEDS[@]}"; do
  one $E $mg $T $s &
  while [ "$(jobs -rp | wc -l)" -ge "$NJOBS" ]; do wait -n 2>/dev/null || sleep 1; done
done; done; done; done
wait
echo "TRACKING_STIFF_COMPLETE"
