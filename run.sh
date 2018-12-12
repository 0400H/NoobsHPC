#!/bin/bash
set -e

unset KMP_AFFINITY
export KMP_AFFINITY="granularity=fine,compact,0,0" # when HT if OFF
# export KMP_AFFINITY="granularity=fine,compact,1,0" # when HT is ON

core_per_socker=`lscpu | grep "Core(s) per socket" | awk -F ':' '{print $2}' | sed 's/^ *\| *$//g'` 
core_num=$core_per_socker
# core_num=1

core_idx=$[$core_num-1]
core_range='0-'$core_idx
echo $core_range

unset OMP_NUM_THREADS
unset MKL_NUM_THREADS
export OMP_NUM_THREADS=$core_num
export MKL_NUM_THREADS=$core_num

cd build && rm -rf log

output_dir=output

taskset -c ${core_range} numactl -l ${output_dir}/unit_test/test_fc_x86

cd -