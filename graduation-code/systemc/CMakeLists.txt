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

function(enable_project_warnings target)
    target_compile_options(${target} PRIVATE
        $<$<CXX_COMPILER_ID:AppleClang,Clang,GNU>:-Wall;-Wextra;-Wpedantic>)
endfunction()

set(SYSTEMC_HOME "$ENV{SYSTEMC_HOME}" CACHE PATH "SystemC installation root")
if(SYSTEMC_HOME)
    list(PREPEND CMAKE_PREFIX_PATH "${SYSTEMC_HOME}")
endif()
find_package(SystemCLanguage CONFIG REQUIRED)

add_executable(atu_hpu_demo src/atu_hpu_demo.cpp)
target_include_directories(atu_hpu_demo PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_link_libraries(atu_hpu_demo PRIVATE SystemC::systemc)
enable_project_warnings(atu_hpu_demo)
enable_project_sanitizers(atu_hpu_demo)
if(SYSTEMC_HOME AND APPLE)
    set_target_properties(
        atu_hpu_demo PROPERTIES BUILD_RPATH "${SYSTEMC_HOME}/lib")
endif()

add_executable(command_codec_tests tests/command_codec_tests.cpp)
target_include_directories(command_codec_tests PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/include")
enable_project_warnings(command_codec_tests)
enable_project_sanitizers(command_codec_tests)

add_library(fp64_reference STATIC reference/src/fp64_reference.cpp)
target_include_directories(fp64_reference PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/reference/include")
enable_project_warnings(fp64_reference)
enable_project_sanitizers(fp64_reference)

add_executable(fp64_reference_tests reference/tests/fp64_reference_tests.cpp)
target_include_directories(fp64_reference_tests PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/third_party")
target_link_libraries(fp64_reference_tests PRIVATE fp64_reference)
enable_project_warnings(fp64_reference_tests)
enable_project_sanitizers(fp64_reference_tests)

add_executable(fp64_reference_smoke reference/tools/fp64_reference_smoke.cpp)
target_link_libraries(fp64_reference_smoke PRIVATE fp64_reference)
enable_project_warnings(fp64_reference_smoke)
enable_project_sanitizers(fp64_reference_smoke)

# ---------------------------------------------------------------------------
# T04 data plane (plain C++ data/validation layer; SystemC tests drive it)
# ---------------------------------------------------------------------------

add_library(dataplane STATIC
    src/dataplane/artifact_loader.cpp
    src/dataplane/device_memory.cpp
    src/dataplane/memory_service.cpp
    src/dataplane/buffer_manager.cpp
    src/dataplane/descriptor_reader.cpp
    src/dataplane/dma_engine.cpp
    src/dataplane/trace.cpp
    src/dataplane/dataplane.cpp
    src/dataplane/selftest.cpp
)
target_include_directories(dataplane
    PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include"
    PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/third_party")
enable_project_warnings(dataplane)
enable_project_sanitizers(dataplane)

add_executable(dataplane_core_tests tests/dataplane_core_tests.cpp)
target_include_directories(dataplane_core_tests PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/third_party")
target_link_libraries(dataplane_core_tests PRIVATE dataplane)
enable_project_warnings(dataplane_core_tests)
enable_project_sanitizers(dataplane_core_tests)

add_executable(dataplane_systemc_tests tests/dataplane_systemc_tests.cpp)
target_include_directories(dataplane_systemc_tests PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/third_party")
target_link_libraries(dataplane_systemc_tests PRIVATE dataplane SystemC::systemc)
enable_project_warnings(dataplane_systemc_tests)
enable_project_sanitizers(dataplane_systemc_tests)
if(SYSTEMC_HOME AND APPLE)
    set_target_properties(
        dataplane_systemc_tests PROPERTIES BUILD_RPATH "${SYSTEMC_HOME}/lib")
endif()

add_executable(dataplane_artifact_tests tests/dataplane_artifact_tests.cpp)
target_include_directories(dataplane_artifact_tests PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/third_party")
target_link_libraries(dataplane_artifact_tests PRIVATE dataplane)
enable_project_warnings(dataplane_artifact_tests)
enable_project_sanitizers(dataplane_artifact_tests)

add_executable(dataplane_smoke src/dataplane_smoke.cpp)
target_link_libraries(dataplane_smoke PRIVATE dataplane)
enable_project_warnings(dataplane_smoke)
enable_project_sanitizers(dataplane_smoke)

include(CTest)
if(BUILD_TESTING)
    add_test(NAME atu_hpu_demo COMMAND atu_hpu_demo)
    add_test(
        NAME command_codec_tests
        COMMAND command_codec_tests
            "${CMAKE_CURRENT_SOURCE_DIR}/../fixtures/command_schema_v2")
    add_test(
        NAME fp64_reference_tests
        COMMAND fp64_reference_tests
            "${CMAKE_CURRENT_SOURCE_DIR}/reference/fixtures/fp64_reference_v1.json"
            "${CMAKE_CURRENT_SOURCE_DIR}/../fixtures/reference_tree/tree_policy_cases.json")

    set(T04_PYTHON "${CMAKE_CURRENT_SOURCE_DIR}/../../.venv/bin/python"
        CACHE FILEPATH "Python interpreter used to generate T04 fixtures")
    if(NOT EXISTS "${T04_PYTHON}")
        find_package(Python3 COMPONENTS Interpreter REQUIRED)
        set(T04_PYTHON "${Python3_EXECUTABLE}")
    endif()
    set(T04_FIXTURE_DIR "${CMAKE_CURRENT_BINARY_DIR}/t04_fixtures")

    add_test(
        NAME t04_generate_fixtures
        COMMAND "${T04_PYTHON}"
            "${CMAKE_CURRENT_SOURCE_DIR}/tools/t04_fixture_gen.py"
            --out "${T04_FIXTURE_DIR}"
            --with-large)

    add_test(NAME dataplane_core_tests COMMAND dataplane_core_tests
        "${T04_FIXTURE_DIR}" "${CMAKE_CURRENT_BINARY_DIR}/t04_output")
    set_tests_properties(dataplane_core_tests PROPERTIES
        DEPENDS t04_generate_fixtures)

    add_test(NAME dataplane_systemc_tests COMMAND dataplane_systemc_tests
        "${T04_FIXTURE_DIR}")
    set_tests_properties(dataplane_systemc_tests PROPERTIES
        DEPENDS t04_generate_fixtures)

    add_test(NAME dataplane_artifact_tests COMMAND dataplane_artifact_tests
        "${T04_FIXTURE_DIR}" "${CMAKE_CURRENT_BINARY_DIR}/t04_output")
    set_tests_properties(dataplane_artifact_tests PROPERTIES
        DEPENDS t04_generate_fixtures)

    add_test(NAME dataplane_smoke COMMAND dataplane_smoke
        --device-dir "${T04_FIXTURE_DIR}/single_static/device"
        --out "${CMAKE_CURRENT_BINARY_DIR}/t04_output/smoke")
    set_tests_properties(dataplane_smoke PROPERTIES
        DEPENDS t04_generate_fixtures)
endif()
