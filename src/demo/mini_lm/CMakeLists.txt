# cmake version
cmake_minimum_required(VERSION 3.22.1)
# project name
project(mini_lm)

set(EXECUTABLE_OUTPUT_PATH ../../../bin)
add_executable(mini_lm ./main.cpp)
target_link_libraries(mini_lm PRIVATE deeplearning)
# GCC < 9 keeps std::filesystem in a separate library
if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9)
  target_link_libraries(mini_lm PRIVATE stdc++fs)
endif()
