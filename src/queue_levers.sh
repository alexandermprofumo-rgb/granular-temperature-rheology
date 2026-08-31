#!/bin/bash
# Start the lever campaign once the mu_g=0 deep dive is finished.
# Keys off the machine going quiet rather than off a driver PID: editing a
# running bash script can make it skip its trailing `wait`, so process-based
# chaining is not reliable here (learned the hard way on the 2D->3D handoff).
set -u
cd "$(dirname "$0")"
q=0
while [ $q -lt 3 ]; do
  sleep 60
  if [ "$(pgrep lmp_serial | wc -l | tr -d ' ')" -eq 0 ]; then q=$((q+1)); else q=0; fi
done
echo "quiet at $(date); sweeping up mu0_deep, then starting levers"
./run_mu0_deep.sh 16
echo "mu0_deep settled at $(date)"
exec ./run_levers_steady.sh 16 all
