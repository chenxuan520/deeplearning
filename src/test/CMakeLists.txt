# cmake version
cmake_minimum_required(VERSION 3.22.1)
# project name
project(test)

# add source
set(EXECUTABLE_OUTPUT_PATH ../../bin)
add_executable(test_bin ./main.cpp ${DIR_SRCS})
target_include_directories(test_bin PRIVATE ../third_party/cpptest)
target_link_libraries(test_bin PRIVATE deeplearning drawtool)
