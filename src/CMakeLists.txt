# cmake version
cmake_minimum_required(VERSION 3.22.1)

if(APPLE AND (NOT CMAKE_OSX_SYSROOT OR NOT EXISTS "${CMAKE_OSX_SYSROOT}"))
	execute_process(
		COMMAND xcrun --show-sdk-path
		OUTPUT_VARIABLE MACOS_SDK_PATH
		OUTPUT_STRIP_TRAILING_WHITESPACE)
	if(MACOS_SDK_PATH)
		set(CMAKE_OSX_SYSROOT "${MACOS_SDK_PATH}" CACHE PATH "" FORCE)
	endif()
endif()

# project name
project(deeplearning)

set(PROJECT_ROOT_PATH .)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# default is Release
set(CMAKE_BUILD_TYPE "Debug")

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

file(GLOB_RECURSE DEEPLEARNING_SOURCES CONFIGURE_DEPENDS
	"${PROJECT_SOURCE_DIR}/deeplearning/*.cpp")

add_library(deeplearning STATIC ${DEEPLEARNING_SOURCES})
target_include_directories(deeplearning PUBLIC ${PROJECT_SOURCE_DIR}/deeplearning)

add_library(drawtool STATIC ${PROJECT_SOURCE_DIR}/drawtool/matplot_draw.cpp)
target_include_directories(drawtool PUBLIC ${PROJECT_SOURCE_DIR}/drawtool)

install(TARGETS deeplearning DESTINATION lib)
install(DIRECTORY ./deeplearning DESTINATION include FILES_MATCHING PATTERN "*.h")

add_subdirectory(${PROJECT_SOURCE_DIR}/test)
add_subdirectory(${PROJECT_SOURCE_DIR}/demo/mnist)
add_subdirectory(${PROJECT_SOURCE_DIR}/demo/transformer_char)
add_subdirectory(${PROJECT_SOURCE_DIR}/demo/optimizer_bench)
