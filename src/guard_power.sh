#!/bin/bash
# Suspend the campaign whenever the machine is on battery; resume on AC.
#
# WHY. Sixteen CPU-bound jobs draw more than the battery should be asked to
# sustain, and an unnoticed unplugged cable turns a long run into a flat
# battery. This makes that failure cost time instead of a shutdown.
#
# SIGSTOP/SIGCONT are idempotent, so this re-asserts the state every cycle
# rather than only on transitions -- that way jobs launched by the driver
# WHILE on battery also get caught, within one poll interval.
#
# On exit (including kill) it always resumes, so the guard dying can never
# leave the campaign permanently frozen.
set -u
INTERVAL=${1:-30}

resume_all(){ pkill -CONT -x lmp_serial 2>/dev/null; }
trap 'echo "$(date "+%F %T") guard exiting - resuming jobs"; resume_all; exit 0' \
     INT TERM EXIT

state=unknown
while true; do
  if pmset -g batt 2>/dev/null | grep -q "AC Power"; then
    pkill -CONT -x lmp_serial 2>/dev/null
    [ "$state" != "ac" ] && echo "$(date '+%F %T')  AC power - running"
    state=ac
  else
    pkill -STOP -x lmp_serial 2>/dev/null
    [ "$state" != "batt" ] && \
      echo "$(date '+%F %T')  ON BATTERY - campaign suspended (will resume on AC)"
    state=batt
  fi
  sleep "$INTERVAL"
done
