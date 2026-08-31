#!/bin/bash
# THE THIRD LEVER ON chi.
#
# chi is the only microstructural candidate not falsified once the window
# systematic is carried: 1.4 sigma, against 2.7-3.5 for the other eight. But
# the friction-vs-stiffness test only has power against discrepancies above
# 0.065 in n over an n-span of 0.143, so "not falsified" is partly just lost
# power. Pressure is orthogonal to both existing levers and spans 10x here,
# and it decides the question.
#
# DESIGN. The friction-lever curve n(chi) already exists at Pconf = 10, with
# chi, from sweep_matched2d + tier1_matched2d_results.csv. This campaign adds
# (n, chi) points at P = 5 and P = 50 -- a 10x lever, the widest the packing
# tolerates (P = 2 is barostat-unstable at the rigid-grain jamming point, and
# every P = 2 cell in the existing scan is VOID). The test is whether the
# P = 5 and P = 50 points fall on the P = 10 friction curve.
#
# Both n and chi come from the SAME runs, which the earlier two-lever tests
# could not do -- they matched n from one sweep to microstructure from
# another. That removes a matching assumption as well as adding a lever.
#
# GRID. 6 frictions x 2 pressures x 6 Theta setpoints x 2 seeds = 144 runs.
# Six setpoints because fit_local needs >= 4 SURVIVING, and the gates
# typically remove one or two (the coldest is I-deviant in nearly every
# sweep). Frictions span mu_g = 0.05-0.5, which is where chi does its moving:
# it collapses 1.00 -> 0.25 over mu_g = 0 -> 0.1.
#
# Scalings (identical to in.granular_2d_piscan -- see the header there):
#   gdot ~ sqrt(P), tdamp ~ 1/sqrt(P), Tgran ~ P, dt ~ 1/sqrt(P)
# so that only kappa = E/P moves.
set -u
cd "$(dirname "$0")"
NJOBS=${1:-14}
mkdir -p sweep_pichi

# seed packings: reuse the pressure-scan ones so the initial states are
# identical to the existing kappa lever
for s in 1 2; do
  [ -f "sweep_pichi/data.pi_seed${s}" ] || cp "sweep_piscan/data.pi_seed${s}" "sweep_pichi/data.pi_seed${s}"
done

launch(){ local lab=$1; shift
  [ -f "sweep_pichi/log.${lab}" ] && grep -q "^DONE" "sweep_pichi/log.${lab}" && return
  ( cd sweep_pichi && lmp_serial "$@" -var label "$lab" > "run.${lab}.out" 2>&1 ) &
  while [ "$(jobs -rp | wc -l)" -ge "$NJOBS" ]; do wait -n 2>/dev/null || sleep 1; done
}

for mg in 0.05 0.1 0.15 0.2 0.3 0.5; do
 for P in 5.0 50.0; do
  for T10 in 0.001 0.0015 0.003 0.005 0.013 0.03; do
   for s in 1 2; do
    read gdot tdamp dt Tg < <(python3 -c "
import math
P=$P; T10=$T10
print(f'{3.162e-4*math.sqrt(P):.6e} {1.581/math.sqrt(P):.6f} {0.001*math.sqrt(10.0/P):.6e} {T10*P/10.0:.6e}')")
    launch "P${P}_mu${mg}_T${T10}_s${s}" -in ../in.granular_2d_pichi \
      -var datafile "data.pi_seed${s}" -var mu_g $mg -var Pconf $P -var gdot $gdot \
      -var tdamp $tdamp -var dt $dt -var Tgran $Tg -var seed $s \
      -var nequil 40000 -var nmeas 80000 -var cdump_every 10000
done; done; done; done

wait
echo "PICHI_COMPLETE"
