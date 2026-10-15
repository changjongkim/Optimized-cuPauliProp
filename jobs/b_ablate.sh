#!/bin/bash
# Row-choice ablation, one loop and one srun at a time on one allocation.
# re: best of eight candidates; rb: one random valid candidate (sampling retained).
# PP_B_JOB reuses an idle pp_ allocation; otherwise request four nodes once.
# PP_B_GPUS=16 (default); PP_B_GPUS="16 4" adds the optional one-node ladder.
set -u -o pipefail
source /pscratch/sd/s/sgkim/kcj/pauliprop/jobs/env.sh
cd "$PP_ROOT" || exit 1
export MPICH_GPU_SUPPORT_ENABLED=0
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
HOSTS=$(scontrol show hostnames "$(squeue -h -j "$JOB" -o %N)" | paste -sd,)
FIRST=${HOSTS%%,*}
printf '%s\n' "$JOB" > logs/b/allocation.id
echo "$(date -u +%FT%TZ) allocation=$JOB loop=$$ hosts=$HOSTS"
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
  local m=$1 c=$2 g=$3 x=$4 n=$(( $3 / 4 )) hosts=$HOSTS candidates=8
  [ "$n" -eq 1 ] && hosts=$FIRST
  [ "$m" = rb ] && candidates=1
  local log="logs/b/$m.$c.g$g.$x.out"
  [ -s "$log" ] && grep -q '^exit=' "$log" && return 0
  [ -e "logs/b/stop.$m.$c.g$g" ] && return 0
  [ "$(squeue -h -j "$JOB" -o %T)" = RUNNING ] || return 90
  # Preserve interrupted logs and retry only on a later, explicit invocation.
  if [ -e "$log" ]; then
    mv "$log" "$log.interrupted.$(date -u +%Y%m%dT%H%M%S)"
  fi
  printf 'allocation=%s mode=%s circuit=%s gpus=%s cutoff=%s candidates=%s\n' "$JOB" "$m" "$c" "$g" "$x" "$candidates" > "$log"
  timeout --signal=TERM --kill-after=30 2400 srun --jobid="$JOB" --nodelist="$hosts" --exclusive --exact \
    -N "$n" -n "$n" --ntasks-per-node=1 -c 128 --gpus-per-node=4 --cpu-bind=none \
    stdbuf -oL ./bin/pp_engine_x --gpus 4 ${CIRC[$c]} --coef "$x" --trace 1 --vram-frac 0.9 --cands "$candidates" >> "$log" 2>&1
  local code=$?
  if [ "$code" -ne 0 ] && [ "$(squeue -h -j "$JOB" -o %T)" != RUNNING ]; then
    echo "allocation ended; incomplete log retained: $log"
    return 90
  fi
  echo "exit=$code" >> "$log"
  [ "$code" -eq 0 ] || touch "logs/b/stop.$m.$c.g$g"
  echo "$(date -u +%FT%TZ) $m $c g$g $x exit=$code $(grep -m1 '^expectation=' "$log") $(grep -m1 '^wall=' "$log")"
  return 0
}
for g in ${PP_B_GPUS:-16}; do
  [[ "$g" == 16 || "$g" == 4 ]] || { echo "only 16 or 4 GPUs supported"; exit 1; }
  for c in ki tfim noise tfimsq xxz qaoa hea; do
    ladder="1e-4 3e-5 1e-5 3e-6"
    [[ "$c" == qaoa || "$c" == hea ]] && ladder="$ladder 1e-6"
    for x in $ladder; do
      for m in re rb; do run "$m" "$c" "$g" "$x" || exit $?; done
    done
  done
done
# The exit trap releases only an allocation created by this invocation.
echo "$(date -u +%FT%TZ) finished"
