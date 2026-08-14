#!/bin/bash
# Block until a the HPC system PBS array finishes, then report per-task status.
# Usage: sims/watch_gadi_sweep.sh <JOBID> [poll_seconds]
JOB="${1:?usage: watch_gadi_sweep.sh <JOBID> [poll_s]}"
POLL="${2:-120}"
while true; do
  # Only trust a POSITIVE response. A dropped SSH returns empty, which an
  # earlier version of this script read as "zero tasks left" and reported a
  # false finish.
  OUT=$(ssh -o BatchMode=yes -o ConnectTimeout=25 <hpc-host> \
        "qstat -tu <username> 2>/dev/null | grep '^${JOB}\['" 2>/dev/null)
  RC=$?
  if [ $RC -ne 0 ] && [ -z "$OUT" ]; then
    # Distinguish "SSH failed" from "job genuinely gone" by asking again for
    # anything at all from qstat.
    PROBE=$(ssh -o BatchMode=yes -o ConnectTimeout=25 <hpc-host> \
            "qstat -B 2>/dev/null | wc -l" 2>/dev/null)
    if ! [[ "$PROBE" =~ ^[0-9]+$ ]] || [ "$PROBE" -lt 2 ]; then
      echo "$(date '+%H:%M:%S')  ssh/qstat unreachable — retrying"
      sleep "$POLL"; continue
    fi
    echo "$(date '+%H:%M:%S')  SWEEP FINISHED (no tasks left in queue)"
    break
  fi
  R=$(echo "$OUT" | grep -c ' R ')
  Q=$(echo "$OUT" | grep -c ' Q ')
  echo "$(date '+%H:%M:%S')  running=$R queued=$Q"
  sleep "$POLL"
done
ssh <hpc-host> "ls -la $PROJECT_ROOT/results_3d_v6/*.pkl 2>/dev/null"
