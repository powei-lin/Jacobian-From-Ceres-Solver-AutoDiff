cmake_minimum_required(VERSION 3.0)

project(jacobianFromAutoDiff)

find_package(Ceres REQUIRED)

add_executable(main main.cpp)
target_link_libraries(main Ceres::ceres)