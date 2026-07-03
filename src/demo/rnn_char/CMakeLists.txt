# cmake version
cmake_minimum_required(VERSION 3.22.1)
# project name
project(rnn_char)

# add source
set(EXECUTABLE_OUTPUT_PATH ../../../bin)
add_executable(rnn_char ./main.cpp)
target_link_libraries(rnn_char PRIVATE deeplearning)
