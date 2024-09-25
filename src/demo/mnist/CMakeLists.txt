# cmake version
cmake_minimum_required(VERSION 3.22.1)
# project name
project(mnist)

# set c++ version
set(CMAKE_CXX_STANDARD 17)

# include dir add,split by<space>
include_directories(../../deeplearning)
include_directories(../../drawtool)

# link dir add
# link_directories()

# add source
set(EXECUTABLE_OUTPUT_PATH ../../../bin)
add_executable(mnist ./main.cpp)

