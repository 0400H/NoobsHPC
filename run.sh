#!/bin/bash
set -e

unset MKL_ENABLE_INSTRUCTIONS
export MKL_ENABLE_INSTRUCTIONS=AVX512_E1
#export MKL_VERBOSE=1

unset KMP_AFFINITY
export KMP_AFFINITY="granularity=fine,compact,0,0" # when HT if OFF
# export KMP_AFFINITY="granularity=fine,compact,1,0" # when HT is ON

core_per_socker=`lscpu | grep "Core(s) per socket" | awk '{print $4}'` 
core_num=$core_per_socker
# core_num=1

core_idx=$[$core_num-1]
core_range='0-'${core_idx}
echo $core_range

unset OMP_NUM_THREADS
unset MKL_NUM_THREADS
export OMP_NUM_THREADS=${core_num}
export MKL_NUM_THREADS=${core_num}

./build.sh

cd build && rm -rf log

output_dir=output

#taskset -c ${core_range} numactl -l ${output_dir}/unit_test/test_gemm
#taskset -c ${core_range} numactl -l ${output_dir}/unit_test/test_convolution
#taskset -c ${core_range} numactl -l ${output_dir}/unit_test/test_inner_product
#taskset -c ${core_range} numactl -l ${output_dir}/unit_test/test_pooling

#taskset -c ${core_range} numactl -l ${output_dir}/benchmark/bench_cblas_gemm_x86
#taskset -c ${core_range} numactl -l ${output_dir}/benchmark/bench_inner_product_x86
#taskset -c ${core_range} numactl -l ${output_dir}/benchmark/bench_convolution_x86

cd -
