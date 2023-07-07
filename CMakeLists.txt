cmake_minimum_required(VERSION 2.6)

project(l2norm)

add_definitions(-std=c++11)

option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Debug)

include_directories(${PROJECT_SOURCE_DIR}/include)
# include and link dirs of cuda and tensorrt, you need adapt them if yours are different
# cuda
include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)


include_directories(/home/xujiahong/TensorRT-7.2.3.4/include/)
link_directories(/home/xujiahong/TensorRT-7.2.3.4/lib/)

# # tensorrt
# include_directories(/usr/include/x86_64-linux-gnu/)
# link_directories(/usr/lib/x86_64-linux-gnu/)

add_executable(l2norm ${PROJECT_SOURCE_DIR}/main.cpp)
target_link_libraries(l2norm nvinfer)
target_link_libraries(l2norm cudart)

add_definitions(-O2 -pthread)