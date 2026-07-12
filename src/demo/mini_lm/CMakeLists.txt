# cmake version
cmake_minimum_required(VERSION 3.22.1)
# project name
project(mini_lm)

set(EXECUTABLE_OUTPUT_PATH ../../../bin)
add_executable(mini_lm ./main.cpp)
target_link_libraries(mini_lm PRIVATE deeplearning)
