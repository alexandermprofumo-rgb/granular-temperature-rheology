#!/bin/bash
# EXTENSION of run_pichi.sh -- the first pass was under-powered.
#
# WHAT THE FIRST PASS SHOWED. The collapse test returned "no failure" at
# 1.3-1.7 sigma, which means nothing: the 2-sigma detectable discrepancy was
# 0.086-0.089 against an n-span of 0.087-0.133, i.e. the test could not resolve
# anything smaller than half the range it was probing. Accepting that as a
# collapse would repeat the exact error that misdirected this project once
# already (an under-powered F-test whose null was read as "negligible").
#
# WHY IT WAS UNDER-POWERED -- and why more seeds alone would NOT have fixed it.
# At P = 5 only 4 of 6 setpoints survive the gates, and fit_local needs 4, so
# leave-one-out leaves 3, the jackknife cannot be formed, and fit_local_sys
# silently falls back to a stat-only error. The P = 5 reference curve was
# therefore quoting understated errors AND could not be jackknifed at all.
#
# The two lost setpoints are the two COLDEST, and they fail on the BAROSTAT:
# sigma_P/P = 0.90 and 0.55 against a 0.15 tolerance. At P = 5 the packing sits
# near the rigid-grain jamming point and the barostat hunts at low temperature
# -- the documented P = 2 pathology in milder form. Adding colder setpoints
# would reproduce the same failure and waste the compute.
#
# SO THIS EXTENSION GOES WARM, NOT COLD:
#   * 3 new setpoints above the current top (T10 = 0.06, 0.12, 0.25), where the
#     barostat is well behaved (sigma_P/P ~ 0.005-0.010 at the current top)
#   * 2 more seeds (3, 4) across the whole grid, halving the statistical term
#
# Per cell this goes from 6 setpoints x 2 seeds to 9 x 4. At P = 5 that should
# give ~7 surviving setpoints -- enough to form the jackknife at last -- and at
# P = 50 a better-constrained fit, where the systematic already dominates in
# 4 of 6 cells.
#
# Expected: total error ~0.022 rather than ~0.044, so a 2-sigma detectable
# discrepancy of ~0.044 against a span of ~0.13. That is a test that can
# actually return a verdict either way.
#
# The gates decide whether the warm setpoints are usable; they are not assumed.
set -u
cd "$(dirname "$0")"
NJOBS=${1:-14}
mkdir -p sweep_pichi

for s in 1 2 3 4; do
  if [ ! -f "sweep_pichi/data.pi_seed${s}" ]; then
    if [ -f "sweep_piscan/data.pi_seed${s}" ]; then
      cp "sweep_piscan/data.pi_seed${s}" "sweep_pichi/data.pi_seed${s}"
    else
      python3 gen_data.py --N 4000 --phi 0.80 --seed $s --out "sweep_pichi/data.pi_seed${s}"
    fi
  fi
done

launch(){ local lab=$1; shift
  [ -f "sweep_pichi/log.${lab}" ] && grep -q "^DONE" "sweep_pichi/log.${lab}" && return
  ( cd sweep_pichi && lmp_serial "$@" -var label "$lab" > "run.${lab}.out" 2>&1 ) &
  while [ "$(jobs -rp | wc -l)" -ge "$NJOBS" ]; do wait -n 2>/dev/null || sleep 1; done
}

# 6 frictions x 2 pressures x 9 setpoints x 4 seeds; already-DONE runs skip.
for mg in 0.05 0.1 0.15 0.2 0.3 0.5; do
 for P in 5.0 50.0; do
  for T10 in 0.001 0.0015 0.003 0.005 0.013 0.03 0.06 0.12 0.25; do
   for s in 1 2 3 4; do
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
echo "PICHI2_COMPLETE"
