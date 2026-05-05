# cmake version
cmake_minimum_required(VERSION 3.22.1)
# project name
project(mnist)

# add source
set(EXECUTABLE_OUTPUT_PATH ../../../bin)
add_executable(mnist ./main.cpp)
target_link_libraries(mnist PRIVATE deeplearning drawtool)
