# cmake version
cmake_minimum_required(VERSION 3.22.1)
# project name
project(optimizer_bench)

# add source
set(EXECUTABLE_OUTPUT_PATH ../../../bin)
add_executable(optimizer_bench ./main.cpp)
target_link_libraries(optimizer_bench PRIVATE deeplearning)
target_include_directories(optimizer_bench PRIVATE ../mnist)
