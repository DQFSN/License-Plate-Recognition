cmake_minimum_required(VERSION 3.13)
project(duanxiaoer)

set(CMAKE_CXX_STANDARD 14)

add_executable(duanxiaoer main.cpp imgProcess.h imgProcess.cpp)

find_package(OpenCV REQUIRED)
target_link_libraries(duanxiaoer ${OpenCV_LIBS})