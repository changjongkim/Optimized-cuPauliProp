#!/bin/bash
# Samples the GPU utilization and memory of this node until it is killed:   g_smi.sh <output prefix> <period in ms>
exec nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used --format=csv,noheader,nounits -lms "$2" > "$1.$(hostname).csv"
