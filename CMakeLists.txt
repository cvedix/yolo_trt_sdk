cmake_minimum_required(VERSION 3.22)
project(yolo_trt_sdk)

# Use ccache to speed up rebuilds
include(cmake/ccache.cmake)

# Set C++ version and optimization level
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Ofast -DNDEBUG -Wno-deprecated-declarations")

# CMake Options
option(ENABLE_BENCHMARKS "Benchmark the preprocessing, inference, and post processing" OFF)
option(BUILD_EXAMPLES "Build example applications" ON)
option(BUILD_TESTS "Build tests" ON)
if (ENABLE_BENCHMARKS)
    add_compile_definitions(ENABLE_BENCHMARKS)
endif()

find_package(OpenCV REQUIRED)

# TODO: Specify the path to TensorRT root dir
set(TensorRT_DIR /home/cyrus/work/libs/TensorRT-10.0.0.6/)

# Build the TensorRT inference engine library
# TensorRT is found and linked by the tensorrt-cpp-api
add_subdirectory(libs/tensorrt-cpp-api)

## Build the SDK library
add_library(${PROJECT_NAME} SHARED src/yolov8.cpp)
add_library(yolo::sdk ALIAS ${PROJECT_NAME})
target_link_libraries(${PROJECT_NAME} PUBLIC tensorrt_cpp_api ${OpenCV_LIBS})
target_include_directories(${PROJECT_NAME}
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/libs/tensorrt-cpp-api/src)

## C bridge facade to simplify embedding in C/C++ callers (NIF, etc.)
add_library(yolo_trt_bridge SHARED src/trt_bridge.cpp)
target_link_libraries(yolo_trt_bridge PUBLIC ${PROJECT_NAME})
target_include_directories(yolo_trt_bridge
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include)

## Examples (optional)
if (BUILD_EXAMPLES)
    add_executable(detect_object_image examples/object_detection_image.cpp)
    target_include_directories(detect_object_image PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/examples)
    target_link_libraries(detect_object_image ${PROJECT_NAME})

    add_executable(benchmark examples/benchmark.cpp)
    target_include_directories(benchmark PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/examples)
    target_link_libraries(benchmark ${PROJECT_NAME})

    add_executable(detect_object_video examples/object_detection_video_stream.cpp)
    target_include_directories(detect_object_video PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/examples)
    target_link_libraries(detect_object_video ${PROJECT_NAME})

    add_executable(detect_object_csi_jetson examples/object_detection_csi_jetson.cpp)
    target_include_directories(detect_object_csi_jetson PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/examples)
    target_link_libraries(detect_object_csi_jetson ${PROJECT_NAME})
endif()

## Tests (optional)
if (BUILD_TESTS)
    enable_testing()
    add_executable(sdk_header_test tests/test_header.cpp)
    target_include_directories(sdk_header_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests)
    target_link_libraries(sdk_header_test ${PROJECT_NAME} ${OpenCV_LIBS})
    add_test(NAME sdk_header_test COMMAND sdk_header_test)
endif()
