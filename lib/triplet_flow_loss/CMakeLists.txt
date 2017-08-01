cmake_minimum_required(VERSION 3.7)
project(triplet_flow_loss)

set(CMAKE_CXX_STANDARD 11)

set(SOURCE_FILES triplet_flow_loss_op.cc)
set(SOURCE_FILES triplet_flow_loss_op_gpu.h)
#add_executable( ${SOURCE_FILES})

add_custom_target(mytest COMMAND ./build.sh CLION_EXE_DIR=${PROJECT_BINARY_DIR})
#add_custom_target(mytest COMMAND echo foo CLION_EXE_DIR=${PROJECT_BINARY_DIR})
