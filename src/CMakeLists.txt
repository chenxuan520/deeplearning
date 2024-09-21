# cmake version
cmake_minimum_required(VERSION 3.22.1)
# project name
project(deeplearning)

set(PROJECT_ROOT_PATH .)

# set check memory
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address -fsanitize=leak -fsanitize=undefined -fno-omit-frame-pointer")
# link_libraries(-fsanitize=address -fsanitize=leak -fsanitize=undefined)

# add matplotlib
option(ENABLE_DRAW "Enable Draw" OFF)
if (ENABLE_DRAW)
	include_directories(./third_party/matplotlib-cpp)
	add_definitions(-D_MATPLOTLIB_CPP_LOAD_)
	add_definitions(-DWITHOUT_NUMPY)

	find_package(PythonInterp REQUIRED)
	find_package(PythonLibs REQUIRED)
	include_directories(${PYTHON_INCLUDE_DIRS})
	link_libraries(${PYTHON_LIBRARIES})
endif()

add_subdirectory(${PROJECT_SOURCE_DIR}/test)
add_subdirectory(${PROJECT_SOURCE_DIR}/demo/mnist)
