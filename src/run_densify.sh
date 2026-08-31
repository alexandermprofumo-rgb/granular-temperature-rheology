#!/bin/bash
# Densify the Theta grid so the exponent can be defined properly.
#
# mu(Theta) is not a power law: the local slope rises with Theta (17/21 cells,
# p=0.0072, shifts up to +0.11). So a single "n" fitted over a window is
# window-dependent, and that systematic (~0.021-0.030) exceeds the statistical
# error and sank most of the quantitative claims.
#
# MORE SETPOINTS DO NOT REMOVE THE CURVATURE -- it is physical. What they do
# is let us FIT it and quote a well-defined local exponent
#     n(Theta_0) = -d ln mu / d ln Theta  evaluated at a common Theta_0
# from a quadratic in log-log. That quantity is unambiguous, and its error
# shrinks with data in the ordinary way.
#
# Existing setpoints {5e-4, 1e-3, 3e-3, 8e-3, 2e-2} (coldest is I-deviant and
# always gated out, so 4 usable). Adding {1.5e-3, 5e-3, 1.3e-2, 3e-2} gives 8
# usable, enough to fit curvature per cell rather than infer it from splits.
#
# Covers the four sweeps the constraint list rests on. Existing runs skipped.
set -u
cd "$(dirname "$0")"
NJOBS=${1:-14}
TNEW=(0.0015 0.005 0.013 0.03)
SEEDS=(1 2 3)
if command -v gtimeout >/dev/null 2>&1; then TO="gtimeout 7200"; else TO="timeout 7200"; fi

launch(){ # $1=dir $2=label $3...=lmp args
  local d=$1 lab=$2; shift 2
  [ -f "$d/log.${lab}" ] && grep -q "^DONE" "$d/log.${lab}" && return
  ( cd "$d" && $TO lmp_serial "$@" -var label "$lab" > "run.${lab}.out" 2>&1 ) &
  while [ "$(jobs -rp | wc -l)" -ge "$NJOBS" ]; do wait -n 2>/dev/null || sleep 1; done
}

# 1. 2D friction curve
for mg in 0.0 0.05 0.1 0.15 0.2 0.3 0.5 1.0; do for T in "${TNEW[@]}"; do for s in "${SEEDS[@]}"; do
  launch sweep_matched2d "mu${mg}_T${T}_s${s}" -in ../in.granular_2d \
     -var datafile "data.m2d_seed${s}" -var mu_g $mg -var gdot 1.0e-3 \
     -var Pconf 10.0 -var Tgran $T -var seed $s -var nequil 40000 -var nmeas 80000
done; done; done

# 2. 3D friction curve (core mu_g only; 3D runs are ~2.5x slower)
for mg in 0.0 0.05 0.1 0.15 0.2 0.3 0.5 1.0; do for T in "${TNEW[@]}"; do for s in "${SEEDS[@]}"; do
  launch sweep_run_3d "mu${mg}_T${T}_s${s}" -in ../in.granular_3d \
     -var datafile "data.granular3d_seed${s}" -var mu_g $mg -var gdot 1.0e-3 \
     -var Pconf 10.0 -var Tgran $T -var seed $s -var nequil 40000 -var nmeas 80000
done; done; done

# 3. stiffness lever
for E in 1.0e4 3.0e4 1.0e5 3.0e5 1.0e6; do
  read dt nr ne nm < <(python3 in.granular_2d_stiff.py $E)
  for mg in 0.1 0.3 1.0; do for T in "${TNEW[@]}"; do for s in "${SEEDS[@]}"; do
    launch sweep_stiffness "E${E}_mu${mg}_T${T}_s${s}" -in ../in.granular_2d_stiff \
       -var datafile "data.stiff_seed${s}" -var mu_g $mg -var gdot 1.0e-3 \
       -var Pconf 10.0 -var Tgran $T -var seed $s -var Emod $E \
       -var dt $dt -var nrelax $nr -var nequil $ne -var nmeas $nm
  done; done; done
done

# 4. tangential-stiffness lever
for kt in 1.0e4 3.0e4 1.0e5 3.0e5; do for mg in 0.1 0.3 1.0; do for T in "${TNEW[@]}"; do for s in "${SEEDS[@]}"; do
  launch sweep_kt_P10 "k${kt}_mu${mg}_T${T}_s${s}" -in ../in.granular_2d_sens \
     -var datafile "data.ktP10_seed${s}" -var mu_g $mg -var gdot 1.0e-3 \
     -var Pconf 10.0 -var Tgran $T -var seed $s -var e_rest 0.5 -var kt_spec $kt \
     -var nequil 40000 -var nmeas 80000
done; done; done; done

wait
echo "DENSIFY_COMPLETE"
