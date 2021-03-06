CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)
set_property(GLOBAL PROPERTY PREDEFINED_TARGETS_FOLDER "CMake")

set(PROJECT_NAME image_classification)
project(${PROJECT_NAME})
set(LIBRARY "${PROJECT_NAME}_lib")

set(CMAKE_CONFIGURATION_TYPES "Debug;Release" CACHE STRING "Configs" FORCE)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Check for c++11 compiler support.
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
elseif(COMPILER_SUPPORTS_CXX0X)
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
else()
   message("Compiler ${CMAKE_CXX_COMPILER} has no C++11 support.")
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE})

find_package(OpenCV 2.4 REQUIRED)
set(LIBRARY_DEPS ${OpenCV_LIBS})

set(LIBRARY_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include")
include_directories(${LIBRARY_INCLUDE_DIR} ${OpenCV_INCLUDE_DIRS})

# Add modules.
add_subdirectory(src)
add_subdirectory(samples)

# Print summary.
message( STATUS "")
message( STATUS "General configuration for ${PROJECT_NAME}")
message( STATUS "======================================")
message( STATUS "")
message( STATUS "   Configuration:        ${CMAKE_BUILD_TYPE}")
message( STATUS "   OpenCV build path:    ${OpenCV_DIR}")
message( STATUS "")