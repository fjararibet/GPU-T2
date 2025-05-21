cmake_minimum_required(VERSION 3.16)
project(GameOfLife)
set(CMAKE_CXX_STANDARD 17)

set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "Disable GLFW examples")
set(GLFW_BUILD_TESTS    OFF CACHE BOOL "Disable GLFW tests")
set(GLFW_BUILD_DOCS     OFF CACHE BOOL "Disable GLFW docs")
set(GLFW_INSTALL        OFF CACHE BOOL "Disable GLFW install")
add_subdirectory(external/glfw)

add_executable(GameOfLife main.cpp)
target_link_libraries(GameOfLife PRIVATE glfw)

