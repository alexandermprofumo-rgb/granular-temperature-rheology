#!/bin/bash
# Open questions 1 and 2.
#
# Q2 -- ARE THE TWO SIGN CHANGES THE SAME CROSSING?
#   The curvature of mu(Theta) changes sign between mu_g = 0.1 and 0.15; the
#   pressure response b(mu_g) = dn/dln(kappa) changes sign between 0.07 and
#   0.1. Two independent quantities changing character at nearly the same
#   friction is the sharpest structural clue left. If they cross at the SAME
#   mu_g, one mechanism controls both and it is worth naming. If they cross
#   at demonstrably different mu_g, they are separate effects.
#   -> densify mu_g in [0.06, 0.25] for the curvature (matched2d, Pconf=10)
#      and in [0.08, 0.09] for b (piscan, all four pressures).
#
# Q1 -- IS n ITSELF THERMOSTAT-DISTORTED?
#   mu is distorted at low friction (+22.7%, 3.2 sigma at mu_g=0.1). Whether
#   n is remains open; no lever was thought to exist because restitution
#   moves Theta only ~30% without a bath. But the bath does not have to be
#   removed, only WEAKENED in a controlled way: n(Pi_damp) measured across a
#   32x range of thermostat coupling, extrapolated toward the weak-bath limit.
#   Pi_damp = t_damp*sqrt(P/rho)/d, so at Pconf=10 a t_damp scan IS a Pi_damp
#   scan. Tier 0-A already showed n depends on it; this measures the
#   dependence properly, with the local exponent and a full setpoint grid.
#   Caveat known in advance: as the bath weakens Theta stops tracking its
#   setpoint, so the gates will eventually reject cells. The point at which
#   they do is itself the answer to "how weak can the bath get before the
#   measurement stops meaning anything".
set -u
cd "$(dirname "$0")"
NJOBS=${1:-14}
TG=(0.001 0.0015 0.003 0.005 0.008 0.013 0.02 0.03)
SEEDS=(1 2 3)
if command -v gtimeout >/dev/null 2>&1; then TO="gtimeout 7200"; else TO="timeout 7200"; fi

launch(){ local d=$1 lab=$2; shift 2
  [ -f "$d/log.${lab}" ] && grep -q "^DONE" "$d/log.${lab}" && return
  ( cd "$d" && $TO lmp_serial "$@" -var label "$lab" > "run.${lab}.out" 2>&1 ) &
  while [ "$(jobs -rp | wc -l)" -ge "$NJOBS" ]; do wait -n 2>/dev/null || sleep 1; done
}

# Q2a: curvature crossing, dense mu_g (2D, Pconf=10)
for mg in 0.06 0.08 0.11 0.12 0.13 0.17 0.25; do for T in "${TG[@]}"; do for s in "${SEEDS[@]}"; do
  launch sweep_matched2d "mu${mg}_T${T}_s${s}" -in ../in.granular_2d \
    -var datafile "data.m2d_seed${s}" -var mu_g $mg -var gdot 1.0e-3 \
    -var Pconf 10.0 -var Tgran $T -var seed $s -var nequil 40000 -var nmeas 80000
done; done; done

# Q2b: b(mu_g) crossing, dense mu_g across all pressures
for mg in 0.08 0.09; do for P in 5.0 10.0 25.0 50.0; do for T10 in 0.001 0.003 0.008 0.02; do for s in "${SEEDS[@]}"; do
  read gdot tdamp dt Tg < <(python3 -c "
import math
P=$P; T10=$T10
print(f'{3.162e-4*math.sqrt(P):.6e} {1.581/math.sqrt(P):.6f} {0.001*math.sqrt(10.0/P):.6e} {T10*P/10.0:.6e}')")
  launch sweep_piscan "P${P}_mu${mg}_T${T10}_s${s}" -in ../in.granular_2d_piscan \
    -var datafile "data.pi_seed${s}" -var mu_g $mg -var Pconf $P -var gdot $gdot \
    -var tdamp $tdamp -var dt $dt -var Tgran $Tg -var seed $s \
    -var nequil 40000 -var nmeas 80000
done; done; done; done

# Q1: thermostat-coupling scan (Pi_damp = tdamp*sqrt(10))
mkdir -p sweep_pidamp
for s in "${SEEDS[@]}"; do
  [ -f "sweep_pidamp/data.pd_seed${s}" ] || python3 gen_data.py --N 4000 --phi 0.80 --seed $s --out "sweep_pidamp/data.pd_seed${s}"
done
for td in 0.125 0.25 0.5 1.0 2.0 4.0; do for mg in 0.1 0.3 1.0; do for T in "${TG[@]}"; do for s in "${SEEDS[@]}"; do
  launch sweep_pidamp "td${td}_mu${mg}_T${T}_s${s}" -in ../in.granular_2d_piscan \
    -var datafile "data.pd_seed${s}" -var mu_g $mg -var Pconf 10.0 -var gdot 1.0e-3 \
    -var tdamp $td -var dt 0.001 -var Tgran $T -var seed $s \
    -var nequil 40000 -var nmeas 80000
done; done; done; done

wait
echo "OPENQ_COMPLETE"
