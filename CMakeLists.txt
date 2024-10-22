cmake_minimum_required(VERSION 3.25)
project(parp_lb3)

set(CMAKE_CXX_STANDARD 20)

add_executable(parp_lb3.exe main.cpp)
target_link_libraries(parp_lb3.exe PRIVATE gomp)
target_compile_options(parp_lb3.exe PRIVATE "-std=c++20 -fopenmp")