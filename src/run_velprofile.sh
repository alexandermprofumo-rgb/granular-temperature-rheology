#!/bin/bash
# THE VELOCITY-PROFILE CHECK.  Every Theta in this project is compute
# temp/deform, which subtracts the AFFINE field v_x = gdot*y. If the real
# profile is not affine, the missing streaming motion is counted as Theta and
# every n inherits the error. No sweep dumps per-atom velocities, so this
# cannot be checked on existing data -- hence twelve fresh cells, identical to
# sweep_steady2d in every other respect (same deck physics, same packings, same
# setpoints), with a binned v_x(y) profile added.
#
# Cells span the range that matters: the coldest and hottest usable setpoint at
# mu_g = 0 (the headline cell), 0.15 (the peak) and 0.3 (where separability
# fails).  ~7 core-hours.
#
# Usage:  ./run_velprofile.sh [njobs]
set -u
cd "$(dirname "$0")"
NJOBS=${1:-14}
D=sweep_velprof; mkdir -p $D
for s in 1 2; do
  [ -f "$D/data.m2d_seed${s}" ] || cp sweep_steady2d/data.m2d_seed${s} "$D/data.m2d_seed${s}"
done
throttle(){ while [ "$(jobs -rp | wc -l)" -ge "$NJOBS" ]; do wait -n 2>/dev/null || sleep 1; done; }
launch(){ local lab=$1; shift
  if [ -f "$D/log.${lab}" ] && grep -q "^DONE" "$D/log.${lab}"; then return; fi
  rm -f "$D/log.${lab}"
  ( cd "$D" && lmp_serial "$@" -var label "$lab" </dev/null > "run.${lab}.out" 2>&1 ) &
  throttle; }
for mg in 0.0 0.15 0.3; do for T in 0.001 0.03; do for s in 1 2; do
  launch "mu${mg}_T${T}_s${s}" -in ../in.granular_2d_ss_prof \
    -var datafile "data.m2d_seed${s}" -var mu_g $mg -var gdot 1.0e-3 \
    -var Pconf 10.0 -var Tgran $T -var seed $s \
    -var nequil 500000 -var nmeas 500000 -var cdump_every 250000
done; done; done
wait
echo "VELPROFILE_COMPLETE"
