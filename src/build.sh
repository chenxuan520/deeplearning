#!/bin/bash

# get argv[0], if not exist, default 0
if [ -z $1 ]; then
    CMAKE_USE_DRAW=false
else
    CMAKE_USE_DRAW=$1
fi

# get argv[1], if not exist, default nil
if [ -z $2 ]; then
    CMAKE_BUILD_TYPE=""
else
    CMAKE_BUILD_TYPE="-DCMAKE_BUILD_TYPE=${2}"
    echo $CMAKE_BUILD_TYPE
fi

mkdir -p ./build
cd build
cmake -DENABLE_DRAW=${CMAKE_USE_DRAW} ${CMAKE_BUILD_TYPE} ..
make
