#!/bin/bash
# Tonight's queue. Waits for AC, then runs each stage under caffeinate at 12
# jobs. Cheapest-first so usable results land early; the I-scan's slow arm and
# the 3D lever are last because they are the two expensive ones.
set -u
cd "$(dirname "$0")"
echo "$(date '+%F %T')  waiting for AC power ..."
until pmset -g batt | grep -q "AC Power"; do sleep 60; done
echo "$(date '+%F %T')  AC detected"
for stage in "run_finite_size.sh|finite size (~4.6h)" \
             "run_restitution.sh|restitution (~5h)" \
             "run_iscan_steady.sh|I-scan + F(I) (~9.4h)" \
             "run_stiff3d.sh|3D stiffness lever (~11.5h)"; do
  s="${stage%%|*}"; n="${stage##*|}"
  echo "$(date '+%F %T')  START  $n"
  caffeinate -dims ./$s 12
  echo "$(date '+%F %T')  DONE   $n"
done
echo "$(date '+%F %T')  ALL COMPLETE"
