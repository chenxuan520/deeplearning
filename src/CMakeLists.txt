# cmake version
cmake_minimum_required(VERSION 3.22.1)
# project name
project(deeplearning)

# set c++ version
set(CMAKE_CXX_STANDARD 17)

# include dir add,split by<space>
include_directories(./util)

# link dir add
# link_directories()

# add source
set(EXECUTABLE_OUTPUT_PATH ../bin)
aux_source_directory(./src DIR_SRCS)
add_executable(deeplearning ./main.cpp ${DIR_SRCS})
