#!/bin/bash
# Runs the command under a profiler on rank 0 only. usage: rank0_wrap.sh <nsys|perf> <out_prefix> <cmd...>
mode=$1; out=$2; shift 2
if [ "${SLURM_PROCID:-0}" = "0" ]; then
  case $mode in
    nsys) exec nsys profile -t cuda,osrt -s cpu --cpuctxsw=none -o "$out" --force-overwrite true "$@" ;;
    perf) exec perf record -F 999 -g -o "$out.perf.data" "$@" ;;
  esac
fi
exec "$@"
