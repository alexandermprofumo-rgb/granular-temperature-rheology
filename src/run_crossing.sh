#!/bin/bash
# Localise the two sign changes well enough to test whether they coincide.
#
# The curvature of mu(Theta) crosses zero at mu_g = 0.091 (+0.052/-0.033); the
# pressure response b(mu_g) crosses somewhere in 0.07-0.09, uncertain to
# ~+/-0.05. They are consistent with being ONE crossing but nowhere near
# resolved. If they are the same, a single mechanism governs how n responds to
# BOTH temperature and pressure -- the first mechanistic handle in the project.
#
# TWO THINGS LIMIT b's PRECISION, both fixed here:
#   1. Only 3 pressures enter each b fit. P=5 has just 4 Theta setpoints and
#      its coldest is barostat-unstable (P = 4.43 +/- 3.94, I off 46%), so 3
#      survive -- below the 4 that fit_local needs for a quadratic, hence
#      VOID. Adding 4 setpoints at every pressure gives P=5 seven usable and
#      restores it, taking b from 3 points to 4.
#   2. With 3 points the fit had dof=1, where residual-scaled errors are
#      meaningless (they produced a spurious 5.7 and 13.7 sigma "sharp
#      transition" that evaporated to ~1 sigma once errors were propagated
#      from the data). Four points give dof=2 and a usable residual check.
#
# The curvature crossing is limited only by scatter, so it gets 2 more seeds
# at the frictions that bracket it.
#
# Expect b's error to roughly halve, localising both crossings to ~+/-0.02 --
# enough to say whether they coincide.
set -u
cd "$(dirname "$0")"
NJOBS=${1:-14}
if command -v gtimeout >/dev/null 2>&1; then TO="gtimeout 7200"; else TO="timeout 7200"; fi

launch(){ local d=$1 lab=$2; shift 2
  [ -f "$d/log.${lab}" ] && grep -q "^DONE" "$d/log.${lab}" && return
  ( cd "$d" && $TO lmp_serial "$@" -var label "$lab" > "run.${lab}.out" 2>&1 ) &
  while [ "$(jobs -rp | wc -l)" -ge "$NJOBS" ]; do wait -n 2>/dev/null || sleep 1; done
}

# A: densify Theta at ALL pressures across the crossing region
for mg in 0.05 0.07 0.08 0.09 0.1 0.12 0.15; do
 for P in 5.0 10.0 25.0 50.0; do
  for T10 in 0.0015 0.005 0.013 0.03; do
   for s in 1 2 3; do
    read gdot tdamp dt Tg < <(python3 -c "
import math
P=$P; T10=$T10
print(f'{3.162e-4*math.sqrt(P):.6e} {1.581/math.sqrt(P):.6f} {0.001*math.sqrt(10.0/P):.6e} {T10*P/10.0:.6e}')")
    launch sweep_piscan "P${P}_mu${mg}_T${T10}_s${s}" -in ../in.granular_2d_piscan \
      -var datafile "data.pi_seed${s}" -var mu_g $mg -var Pconf $P -var gdot $gdot \
      -var tdamp $tdamp -var dt $dt -var Tgran $Tg -var seed $s \
      -var nequil 40000 -var nmeas 80000
done; done; done; done

# B: two more seeds on the curvature crossing (2D, Pconf=10)
for s in 4 5; do
  [ -f "sweep_matched2d/data.m2d_seed${s}" ] || python3 gen_data.py --N 4000 --phi 0.80 --seed $s --out "sweep_matched2d/data.m2d_seed${s}"
done
for mg in 0.06 0.08 0.1 0.11 0.12 0.13 0.15; do
 for T in 0.001 0.0015 0.003 0.005 0.008 0.013 0.02 0.03; do
  for s in 4 5; do
   launch sweep_matched2d "mu${mg}_T${T}_s${s}" -in ../in.granular_2d \
     -var datafile "data.m2d_seed${s}" -var mu_g $mg -var gdot 1.0e-3 \
     -var Pconf 10.0 -var Tgran $T -var seed $s -var nequil 40000 -var nmeas 80000
done; done; done

wait
echo "CROSSING_COMPLETE"
