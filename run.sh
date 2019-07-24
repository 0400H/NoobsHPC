#!/bin/bash
set -e

unset MKL_ENABLE_INSTRUCTIONS
export MKL_ENABLE_INSTRUCTIONS=AVX512_E1
#export MKL_VERBOSE=1

unset KMP_AFFINITY
export KMP_AFFINITY="granularity=fine,compact,0,0" # when HT if OFF
# export KMP_AFFINITY="granularity=fine,compact,1,0" # when HT is ON

total_cores=$(lscpu | grep 'CPU(s):' | head -1 | awk '{print $2}')
core_per_socker=$(lscpu | grep "Core(s) per socket" | awk '{print $4}')
core_num=$total_cores
# core_num=$core_per_socker
# core_num=1

core_idx=$[$core_num-2]
core_range='0-'${core_idx}
echo $core_range

unset OMP_NUM_THREADS
unset MKL_NUM_THREADS
export OMP_NUM_THREADS=${core_num}
export MKL_NUM_THREADS=${core_num}

./build.sh

output_dir=./build/output

taskset -c ${core_range} numactl -l ${output_dir}/unit_test/test_axpy
#taskset -c ${core_range} numactl -l ${output_dir}/unit_test/test_gemm_mkl
#taskset -c ${core_range} numactl -l ${output_dir}/unit_test/test_convolution

#taskset -c ${core_range} numactl -l ${output_dir}/benchmark/bench_axpy
#taskset -c ${core_range} numactl -l ${output_dir}/benchmark/bench_gemm_mkl
#taskset -c ${core_range} numactl -l ${output_dir}/benchmark/bench_convolution
