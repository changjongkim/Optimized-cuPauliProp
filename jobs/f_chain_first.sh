#!/bin/bash
# Waits for the drive that was started by hand on an allocation, then hands over to the unattended loop.
#   nohup setsid bash jobs/f_chain_first.sh <allocation id> > logs/f/loop.log 2>&1 &
cd /pscratch/sd/s/sgkim/kcj/pauliprop
until grep -q "drive job=$1 finished" queue_f/runner.log 2>/dev/null; do sleep 20; done
exec bash jobs/f_loop.sh $1
