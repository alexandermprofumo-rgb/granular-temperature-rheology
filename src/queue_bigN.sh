#!/bin/bash
set -u
cd "$(dirname "$0")"
until grep -q "ALL COMPLETE" queued_campaigns.log 2>/dev/null; do sleep 120; done
echo "$(date '+%F %T')  main queue done - starting large-N frictionless"
caffeinate -dims ./run_bigN.sh 12
echo "$(date '+%F %T')  BIGN DONE"
