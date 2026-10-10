#!/bin/bash
# Runs u7 in an interactive allocation and releases it:   bash jobs/u7_drive.sh <allocation id> [defer3]
cd /pscratch/sd/s/sgkim/kcj/pauliprop
bash jobs/u7_breakdown.sh $1 ${2:-} > logs/u7/driver-$1.log 2>&1
scancel $1
