# cmake version
cmake_minimum_required(VERSION 3.22.1)
# project name
project(web_model_export)

set(EXECUTABLE_OUTPUT_PATH ../../../bin)
add_executable(web_model_export ./main.cpp)
target_link_libraries(web_model_export PRIVATE deeplearning)
