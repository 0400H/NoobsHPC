#!/bin/bash
set -e

mkdir -p build && cd build

clear && clear && clear

cmake ..

core_num=`cat /proc/cpuinfo| grep "processor"| wc -l`

make -j $[$core_num-1]
# make doc -j $[$core_num-1]

if [ $? -eq 0 ]; then
    echo "build success"
else
    echo "build fail"
    exit -1
fi

cd -