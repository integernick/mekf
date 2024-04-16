cmake_minimum_required(VERSION 3.21)
project(mekf)

set(CMAKE_CXX_STANDARD 20)

find_package(Eigen3 3.3 REQUIRED NO_MODULE)

add_library(mekf mekf.cpp)
target_include_directories(mekf PUBLIC mathlib)
target_link_libraries(mekf PUBLIC Eigen3::Eigen)
