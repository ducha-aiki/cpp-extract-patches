cmake_minimum_required(VERSION 2.8.12)
project(extract_patches)

add_definitions(-fPIC)
add_definitions(-DA64)
add_definitions(-std=c++14)

SET(CMAKE_BUILD_TYPE "RELEASE")

IF(CMAKE_COMPILER_IS_GNUCXX)
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wno-strict-aliasing")
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wextra -Wno-write-strings")
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated -ansi")
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -ftree-vectorize -funroll-loops")
ENDIF(CMAKE_COMPILER_IS_GNUCXX)

FIND_PACKAGE(OpenMP)
if (USE_OPENMP)
	if(NOT OPENMP_FOUND)
		message(FATAL_ERROR "OPENMP not found.")
	endif()
	add_definitions(-DUSE_OPENMP)
endif (USE_OPENMP)

# ==============================================================================
# Find OpenCV
# ==============================================================================
find_package(OpenCV)
if (OPENCV_FOUND)
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenCV_CXX_FLAGS}")
endif (OPENCV_FOUND)

####### sources
include_directories(include/)

#find_package(Torch REQUIRED)
#include_directories(/home/old-ufo/dev/libtorch/libtorch/include/)

add_executable(extract_patches_example
		src/main.cpp)

target_link_libraries(extract_patches_example ${OpenCV_LIBS})
