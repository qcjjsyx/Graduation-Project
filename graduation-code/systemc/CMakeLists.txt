cmake_minimum_required(VERSION 3.16)

project(graduation_systemc_models LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
option(ENABLE_SANITIZERS "Enable AddressSanitizer and UndefinedBehaviorSanitizer" OFF)

function(enable_project_sanitizers target)
    if(ENABLE_SANITIZERS AND CMAKE_CXX_COMPILER_ID MATCHES "AppleClang|Clang|GNU")
        target_compile_options(${target} PRIVATE
            -fsanitize=address,undefined
            -fno-omit-frame-pointer
        )
        target_link_options(${target} PRIVATE -fsanitize=address,undefined)
    endif()
endfunction()

set(SYSTEMC_HOME "$ENV{SYSTEMC_HOME}" CACHE PATH "SystemC installation root")
if(SYSTEMC_HOME)
    list(PREPEND CMAKE_PREFIX_PATH "${SYSTEMC_HOME}")
endif()
find_package(SystemCLanguage CONFIG REQUIRED)

add_executable(atu_hpu_demo src/atu_hpu_demo.cpp)
target_include_directories(atu_hpu_demo PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_link_libraries(atu_hpu_demo PRIVATE SystemC::systemc)
target_compile_options(atu_hpu_demo PRIVATE
    $<$<CXX_COMPILER_ID:AppleClang,Clang,GNU>:-Wall;-Wextra;-Wpedantic>)
enable_project_sanitizers(atu_hpu_demo)
if(SYSTEMC_HOME AND APPLE)
    set_target_properties(
        atu_hpu_demo PROPERTIES BUILD_RPATH "${SYSTEMC_HOME}/lib")
endif()

add_executable(command_codec_tests src/command_codec_tests.cpp)
target_include_directories(command_codec_tests PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_compile_options(command_codec_tests PRIVATE
    $<$<CXX_COMPILER_ID:AppleClang,Clang,GNU>:-Wall;-Wextra;-Wpedantic>)
enable_project_sanitizers(command_codec_tests)

add_library(fp64_reference STATIC reference/src/fp64_reference.cpp)
target_include_directories(fp64_reference PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/reference/include")
target_compile_options(fp64_reference PRIVATE
    $<$<CXX_COMPILER_ID:AppleClang,Clang,GNU>:-Wall;-Wextra;-Wpedantic>)
enable_project_sanitizers(fp64_reference)

add_executable(fp64_reference_tests reference/tests/fp64_reference_tests.cpp)
target_include_directories(fp64_reference_tests PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/third_party")
target_link_libraries(fp64_reference_tests PRIVATE fp64_reference)
target_compile_options(fp64_reference_tests PRIVATE
    $<$<CXX_COMPILER_ID:AppleClang,Clang,GNU>:-Wall;-Wextra;-Wpedantic>)
enable_project_sanitizers(fp64_reference_tests)

include(CTest)
if(BUILD_TESTING)
    add_test(NAME atu_hpu_demo COMMAND atu_hpu_demo)
    add_test(
        NAME command_codec_tests
        COMMAND command_codec_tests
            "${CMAKE_CURRENT_SOURCE_DIR}/../fixtures/command_schema_v1")
    add_test(
        NAME fp64_reference_tests
        COMMAND fp64_reference_tests
            "${CMAKE_CURRENT_SOURCE_DIR}/reference/fixtures/fp64_reference_v1.json")
endif()
