# cmake version
cmake_minimum_required(VERSION 3.22.1)
# project name
project(rl_tictactoe)

set(EXECUTABLE_OUTPUT_PATH ../../../bin)
add_executable(rl_tictactoe ./main.cpp)
target_link_libraries(rl_tictactoe PRIVATE deeplearning)
