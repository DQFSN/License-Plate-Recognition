cmake_minimum_required(VERSION 3.12)
project(CarID)

set(CMAKE_CXX_STANDARD 14)

add_executable(CarID main.cpp)

find_package(OpenCV REQUIRED)
target_link_libraries(CarID ${OpenCV_LIBS})