#!/bin/bash
# Samples the CPU time counters of this node until it is killed:   g_cpu.sh <output prefix> <period in seconds>
# One line per sample: time, then the first line of /proc/stat (user nice system idle iowait irq softirq ...).
while :; do echo "$(date +%s.%N) $(head -1 /proc/stat)"; sleep "$2"; done > "$1.$(hostname).cpu"
