cmake_minimum_required(VERSION 3.16)
project(GameOfLife)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_FLAGS "-g -Wall -Wextra -pedantic -std=c++17 -Wshadow -Wformat=2 -Wfloat-equal -Wconversion -D_GLIBCXX_DEBUG -D_GLIBCXX_DEBUG_PEDANTIC")

# GLFW
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "Disable GLFW examples")
set(GLFW_BUILD_TESTS    OFF CACHE BOOL "Disable GLFW tests")
set(GLFW_BUILD_DOCS     OFF CACHE BOOL "Disable GLFW docs")
set(GLFW_INSTALL        OFF CACHE BOOL "Disable GLFW install")
add_subdirectory(external/glfw)

# ImGui
set(IMGUI_DIR ${CMAKE_SOURCE_DIR}/external/imgui)
file(GLOB IMGUI_SOURCES
    ${IMGUI_DIR}/*.cpp
    ${IMGUI_DIR}/backends/imgui_impl_glfw.cpp
    ${IMGUI_DIR}/backends/imgui_impl_opengl3.cpp
)
add_library(imgui STATIC ${IMGUI_SOURCES})
target_include_directories(imgui PUBLIC
    ${IMGUI_DIR}
    ${IMGUI_DIR}/backends
)

# OpenCL
find_package(OpenCL REQUIRED)

# Your sources
file(GLOB SOURCES "src/*.cpp" "src/*.c")
add_executable(GameOfLife ${SOURCES})

# Includes & linking
target_include_directories(GameOfLife PRIVATE include/ ${OpenCL_INCLUDE_DIRS})
target_link_libraries(GameOfLife PRIVATE glfw imgui ${OpenCL_LIBRARIES})

