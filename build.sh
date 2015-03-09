#!/bin/bash
set -e

rm -rf noobsdnn_config.h
mkdir -p ./build && cd ./build

cmake ..
make -j `nproc`

if [ $? -eq 0 ]; then
    echo "build success"
else
    echo "build fail"
    exit -1
fi

# ctest ./test --output-on-failure
cd -
