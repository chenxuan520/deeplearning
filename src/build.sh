#!/bin/bash

# get argv[0], if not exist, default 0
if [ -z $1 ]; then
    CMAKE_BUILD_TYPE=false
else
    CMAKE_BUILD_TYPE=$1
fi

mkdir -p ./build
cd build
cmake -DENABLE_DRAW=${CMAKE_BUILD_TYPE} ..
make
