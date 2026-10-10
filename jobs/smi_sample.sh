#!/bin/bash
# Samples the GPU utilization of this node every 200 ms until it is killed:   smi_sample.sh <output prefix>
exec nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used --format=csv,noheader,nounits -lms 200 > "$1.$(hostname).csv"
