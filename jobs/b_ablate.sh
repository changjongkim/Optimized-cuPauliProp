#!/bin/bash
# Row-choice ablation, one loop and one srun at a time on one allocation.
# re: best of eight candidates; rb: one random valid candidate (sampling retained).
# ov1/ov0: overlapped/serial row selection, three repetitions per setting.
# PP_B_JOB reuses an idle pp_ allocation; otherwise request four nodes once.
# PP_B_GPUS=16 (default); PP_B_GPUS="16 4" adds the optional one-node ladder.
# PP_B_DRY_RUN=1 prints the finite run list without requesting an allocation.
set -u -o pipefail
source /pscratch/sd/s/sgkim/kcj/pauliprop/jobs/env.sh
cd "$PP_ROOT" || exit 1
export MPICH_GPU_SUPPORT_ENABLED=0
for g in ${PP_B_GPUS:-16}; do
  [[ "$g" == 16 || "$g" == 4 ]] || { echo "only 16 or 4 GPUs supported"; exit 1; }
done
balance_plan() {
  local x=$1 g c m
  for g in ${PP_B_GPUS:-16}; do
    for c in ki tfim noise tfimsq xxz qaoa hea; do
      [[ "$x" == 1e-6 && "$c" != qaoa && "$c" != hea ]] && continue
      for m in re rb; do printf '%s\t%s\t%s\t%s\t0\n' "$m" "$c" "$g" "$x"; done
    done
  done
}
plan() {
  local rep c x m order
  # Complete the smaller balance pairs and the overlap controls before the
  # expensive cutoff tail, so one slow configuration cannot hide both effects.
  for x in 1e-4 3e-5 1e-5; do balance_plan "$x"; done
  for rep in 1 2 3; do
    order="ov1 ov0"
    [ "$rep" -eq 2 ] && order="ov0 ov1"
    for c in ki tfim noise; do
      for x in 1e-4 1e-5; do
        for m in $order; do printf '%s\t%s\t16\t%s\t%s\n' "$m" "$c" "$x" "$rep"; done
      done
    done
  done
  for x in 3e-6 1e-6; do balance_plan "$x"; done
}
if [ "${PP_B_DRY_RUN:-0}" = 1 ]; then plan; exit 0; fi
mkdir -p logs/b
exec 9>logs/b/loop.lock
flock -n 9 || { echo "another b ablation loop holds the lock"; exit 1; }
printf '%s\n' "$$" > logs/b/loop.pid
JOB="${PP_B_JOB:-}"
OWNED=0
cleanup() {
  local code=$?
  trap - EXIT
  # An error or signal must not leave an allocation created here idle.
  if [ "$OWNED" -eq 1 ] && [[ "$JOB" =~ ^[0-9]+$ ]] &&
     [ "$(squeue -h -j "$JOB" -o %j)" = pp_b ]; then
    scancel "$JOB"
  fi
  if [ "$(cat logs/b/loop.pid 2>/dev/null)" = "$$" ]; then
    rm -f logs/b/loop.pid
  fi
  exit "$code"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
if [ -z "$JOB" ]; then
  # Fail promptly if interactive capacity is unavailable; no retry/wait loop.
  out=$(salloc --immediate=30 -A m5320_g -C "gpu&hbm40g" -q interactive -N 4 -t 04:00:00 --gpus-per-node=4 --no-shell -J pp_b 2>&1)
  rc=$?
  printf '%s\n' "$out"
  [ "$rc" -eq 0 ] || exit "$rc"
  JOB=$(printf '%s\n' "$out" | sed -n 's/.*Granted job allocation \([0-9]*\).*/\1/p' | tail -1)
  OWNED=1
fi
[[ "$JOB" =~ ^[0-9]+$ ]] || { echo "invalid allocation id"; exit 1; }
read -r name state nodes < <(squeue -h -j "$JOB" -o '%j %T %D')
[[ "${name:-}" == pp_* && "${state:-}" == RUNNING && "${nodes:-0}" == 4 ]] || {
  echo "requires a running four-node pp_ allocation"; exit 1;
}
exec 8>"logs/b/allocation.$JOB.lock"
flock -n 8 || { echo "allocation $JOB already has a b loop"; exit 1; }
steps=$(squeue -h --steps -j "$JOB" -o '%i' | sed '/\.extern$/d; /\.batch$/d')
[ -z "$steps" ] || { echo "allocation $JOB already has active steps: $steps"; exit 1; }
BIN_SHA=$(sha256sum bin/pp_engine_x | awk '{print $1}')
SOURCE_COMMIT=$(git rev-parse HEAD)
[ -x bin/pp_engine_x ] && [ -n "$BIN_SHA" ] || { echo "missing engine binary"; exit 1; }
HOSTS=$(scontrol show hostnames "$(squeue -h -j "$JOB" -o %N)" | paste -sd,)
FIRST=${HOSTS%%,*}
printf '%s\n' "$JOB" > logs/b/allocation.id
echo "$(date -u +%FT%TZ) allocation=$JOB loop=$$ hosts=$HOSTS"
printf 'binary_sha256=%s source_commit=%s\n' "$BIN_SHA" "$SOURCE_COMMIT"
plan > "logs/b/plan.$JOB.tsv"
tr '\0' ' ' < "/proc/$$/cmdline"
echo
declare -A CIRC=(
  [ki]="--circuit ki_lab --weight 0 --target 6000000"
  [tfim]="--circuit tfim_hh --theta 0.2 --phi -0.3 --steps 20 --target 4000000"
  [noise]="--circuit ki_noisy --noise 0.005 --target 6000000"
  [tfimsq]="--circuit tfim_sq --lx 11 --ly 11 --theta 0.2 --phi -0.3 --steps 10 --target 4000000"
  [xxz]="--circuit xxz --lx 8 --ly 8 --jxy 0.1 --jz 0.15 --steps 6 --target 4000000"
  [qaoa]="--circuit qaoa --n 100 --layers 4 --target 4000000"
  [hea]="--circuit hea --n 100 --layers 16 --amp 0.4 --target 4000000"
)
run() {
  local m=$1 c=$2 g=$3 x=$4 rep=$5 n=$(( $3 / 4 )) hosts=$HOSTS candidates=8 overlap=1 limit=2400
  [ "$n" -eq 1 ] && hosts=$FIRST
  [ "$m" = rb ] && candidates=1
  [ "$m" = ov0 ] && overlap=0
  local log="logs/b/$m.$c.g$g.$x.out"
  if [ "$rep" -gt 0 ]; then log="logs/b/$m.$c.g$g.$x.r$rep.out"; limit=300; fi
  # Every invocation remeasures the controls in its current allocation.
  # Preserve previous completed and incomplete attempts under unique names.
  [ "$(squeue -h -j "$JOB" -o %T)" = RUNNING ] || return 90
  if [ -e "$log" ]; then
    mv -- "$log" "$log.previous.$JOB.$(date -u +%Y%m%dT%H%M%S)" || return 91
  fi
  local args=(--gpus 4 ${CIRC[$c]} --coef "$x" --trace 1 --vram-frac 0.9 --seed 7 --cands "$candidates" --overlap-plan "$overlap")
  printf 'allocation=%s mode=%s circuit=%s gpus=%s cutoff=%s candidates=%s overlap=%s repeat=%s\n' "$JOB" "$m" "$c" "$g" "$x" "$candidates" "$overlap" "$rep" > "$log"
  printf 'binary=pp_engine_x\nbinary_sha256=%s\nsource_commit=%s\n' "$BIN_SHA" "$SOURCE_COMMIT" >> "$log"
  printf 'args:' >> "$log"
  printf ' %q' "${args[@]}" >> "$log"
  printf '\n' >> "$log"
  timeout --signal=TERM --kill-after=30 "$limit" srun --jobid="$JOB" --nodelist="$hosts" --exclusive --exact \
    -N "$n" -n "$n" --ntasks-per-node=1 -c 128 --gpus-per-node=4 --cpu-bind=none \
    stdbuf -oL ./bin/pp_engine_x "${args[@]}" >> "$log" 2>&1
  local code=$?
  if [ "$code" -ne 0 ] && [ "$(squeue -h -j "$JOB" -o %T)" != RUNNING ]; then
    echo "allocation ended; incomplete log retained: $log"
    return 90
  fi
  echo "exit=$code" >> "$log"
  echo "$(date -u +%FT%TZ) $m $c g$g $x repeat=$rep exit=$code $(grep -m1 '^expectation=' "$log") $(grep -m1 '^wall=' "$log")"
  local remaining
  remaining=$(squeue -h --steps -j "$JOB" -o '%i' | sed '/\.extern$/d; /\.batch$/d')
  [ -z "$remaining" ] || { echo "unfinished step after run: $remaining; stopping allocation"; return 93; }
  # Do not classify a timeout/failed run as successful or infer failures for
  # unexecuted settings. Each requested configuration has its own bounded run.
  if [ "$code" -eq 0 ]; then
    grep -q '^expectation=' "$log" && grep -q '^wall=' "$log" || return 92
  fi
  return 0
}
while IFS=$'\t' read -r mode circuit gpus cutoff repetition; do
  run "$mode" "$circuit" "$gpus" "$cutoff" "$repetition" || exit $?
done < "logs/b/plan.$JOB.tsv"
python3 analysis/paper_n2.py --allocation "$JOB" --write || exit $?
# The exit trap releases only an allocation created by this invocation.
echo "$(date -u +%FT%TZ) finished"
