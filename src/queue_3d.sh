#!/bin/bash
# Start 3D once 2D is genuinely finished.
#
# Robust to the 2D driver exiting early: run_steady_campaign.sh was edited in
# place while an instance of it was running, and bash reads scripts
# incrementally, so its trailing `wait` may be skipped. So do not key off the
# driver process. Instead wait for the machine to go quiet, then re-run the 2D
# stage once -- it is resumable, so it relaunches anything that failed or was
# never started, and is a no-op if everything finished.
set -u
cd "$(dirname "$0")"

quiet=0
while [ $quiet -lt 3 ]; do
  sleep 60
  if [ "$(pgrep lmp_serial | wc -l | tr -d ' ')" -eq 0 ]; then
    quiet=$((quiet + 1))
  else
    quiet=0
  fi
done

echo "machine quiet at $(date); 2D done = $(grep -l '^DONE' sweep_steady2d/log.mu* 2>/dev/null | wc -l)/180"
echo "sweeping up any unfinished 2D runs ..."
./run_steady_campaign.sh 16 2d
echo "2D final = $(grep -l '^DONE' sweep_steady2d/log.mu* 2>/dev/null | wc -l)/180 at $(date)"
echo "starting 3D at $(date)"
exec ./run_steady_campaign.sh 16 3d
