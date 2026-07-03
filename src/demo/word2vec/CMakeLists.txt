# cmake version
cmake_minimum_required(VERSION 3.22.1)
# project name
project(word2vec)

set(EXECUTABLE_OUTPUT_PATH ../../../bin)
add_executable(word2vec ./main.cpp)
target_link_libraries(word2vec PRIVATE deeplearning)
