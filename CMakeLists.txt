cmake_minimum_required(VERSION 3.21)

project(vxray C CXX)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(vxray ${CMAKE_SOURCE_DIR}/src/vxray.c)
