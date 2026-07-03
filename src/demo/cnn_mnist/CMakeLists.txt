# cmake version
cmake_minimum_required(VERSION 3.22.1)
# project name
project(cnn_mnist)

# add source
set(EXECUTABLE_OUTPUT_PATH ../../../bin)
add_executable(cnn_mnist ./main.cpp)
target_link_libraries(cnn_mnist PRIVATE deeplearning)
target_include_directories(cnn_mnist PRIVATE ../mnist)
