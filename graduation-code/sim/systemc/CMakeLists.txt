cmake_minimum_required(VERSION 3.16)

project(graduation_systemc_models LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(atu_hpu_demo
    src/main.cpp
)

target_include_directories(atu_hpu_demo PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Point SYSTEMC_HOME to your SystemC installation if CMake cannot find it.
# Example:
#   cmake -S . -B build -DSYSTEMC_HOME=C:/systemc
set(SYSTEMC_HOME "$ENV{SYSTEMC_HOME}" CACHE PATH "SystemC installation root")

if(SYSTEMC_HOME)
    target_include_directories(atu_hpu_demo PRIVATE "${SYSTEMC_HOME}/include")
    find_library(SYSTEMC_LIBRARY
        NAMES systemc libsystemc
        PATHS
            "${SYSTEMC_HOME}/lib"
            "${SYSTEMC_HOME}/lib64"
            "${SYSTEMC_HOME}/lib-linux64"
            "${SYSTEMC_HOME}/msvc10/SystemC/Debug"
            "${SYSTEMC_HOME}/msvc10/SystemC/Release"
        NO_DEFAULT_PATH
    )
else()
    find_library(SYSTEMC_LIBRARY NAMES systemc libsystemc)
endif()

if(NOT SYSTEMC_LIBRARY)
    message(FATAL_ERROR "SystemC library not found. Set SYSTEMC_HOME or install SystemC in a standard location.")
endif()

target_link_libraries(atu_hpu_demo PRIVATE "${SYSTEMC_LIBRARY}")
