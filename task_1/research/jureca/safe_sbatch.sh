#!/bin/bash
# Team-safe sbatch for <project> / team folder <team dir>.
#   safe_sbatch.sh general <script.sbatch> [sbatch args]   general dc-gpu queue, never inside <reservation>
#   safe_sbatch.sh res     <script.sbatch> [sbatch args]   <reservation> reservation, only if the team holds NO job there
# Rule: the team may hold at most ONE job (running or pending) in <reservation>. <reservation> is MAGNETIC, so a job without
# --reservation can still be pulled into it: general jobs get --exclude=<reservation nodes> and are re-checked after
# submission (cancelled if they landed in the reservation anyway while the team already holds a job).
set -euo pipefail
TEAM=${TEAM_DIR:?set TEAM_DIR}
mode=${1:?general|res}; script=${2:?sbatch script}; shift 2
case "$(pwd)" in "$TEAM"/*) ;; *) echo "refusing: submit from inside $TEAM" >&2; exit 1;; esac
team_jobs() { squeue -r --reservation=${RESERVATION:?set RESERVATION} -h -t PD,R,CF,CG -o "%i|%u|%j|%T|%Z" | grep "$TEAM" || true; }
if [[ $mode == res ]]; then
  busy=$(team_jobs)
  if [[ -n $busy ]]; then echo "refusing: team already holds a ${RESERVATION:?set RESERVATION} job:"; echo "$busy"; exit 2; fi
  job=$(sbatch --parsable --reservation=${RESERVATION:?set RESERVATION} "$@" "$script")
  echo "submitted $job in ${RESERVATION:?set RESERVATION}"
  sleep 2; others=$(team_jobs | grep -v "^$job|" || true)
  if [[ -n $others ]]; then scancel "$job"; echo "race: another team job appeared, cancelled $job:"; echo "$others"; exit 3; fi
else
  nodes=$(scontrol show res ${RESERVATION:?set RESERVATION} | grep -oE " Nodes=[^ ]+" | cut -d= -f2)
  job=$(sbatch --parsable --exclude="$nodes" "$@" "$script")
  sleep 2; res=$(squeue -h -j "$job" -o "%v")
  if [[ $res == ${RESERVATION:?set RESERVATION} ]]; then   # user rule (22/11): heavy general jobs must never occupy the reservation
    scancel "$job"; echo "cancelled $job: it was pulled into ${RESERVATION:?set RESERVATION} despite --exclude"; exit 3
  fi
  echo "submitted $job (reservation: ${res:-none})"
fi
