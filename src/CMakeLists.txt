# cmake version
cmake_minimum_required(VERSION 3.22.1)
# project name
project(deeplearning)

set(PROJECT_ROOT_PATH .)

add_subdirectory(${PROJECT_SOURCE_DIR}/test)
add_subdirectory(${PROJECT_SOURCE_DIR}/demo/mnist)
