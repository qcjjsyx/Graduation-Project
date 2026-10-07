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
# SCALE_DESC policy decoding lives in the dataplane layer (descriptor
# semantics), so the codec test links it explicitly.
target_link_libraries(command_codec_tests PRIVATE dataplane)
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
    src/dataplane/on_chip_arbiter.cpp
    src/dataplane/buffer_manager.cpp
    src/dataplane/scale_desc.cpp
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

# ---------------------------------------------------------------------------
# T05 control plane (command fetch / scoreboard / scheduling / completion)
# ---------------------------------------------------------------------------

add_library(controlplane STATIC
    src/controlplane/command_fetch.cpp
    src/controlplane/token_scoreboard.cpp
    src/controlplane/completion_writer.cpp
    src/controlplane/dma_arbiter.cpp
    src/controlplane/panel_exec_unit.cpp
    src/controlplane/assembly_exec_unit.cpp
    src/controlplane/test_stub_unit.cpp
    src/controlplane/control_plane.cpp
)
target_include_directories(controlplane
    PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include"
    PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/third_party")
target_link_libraries(controlplane PUBLIC dataplane panel_lu)
enable_project_warnings(controlplane)
enable_project_sanitizers(controlplane)

add_library(tree_artifact_checker STATIC reference/src/tree_artifact_check.cpp)
target_include_directories(tree_artifact_checker PUBLIC reference/include PRIVATE third_party)
target_link_libraries(tree_artifact_checker PUBLIC fp64_reference dataplane)
enable_project_warnings(tree_artifact_checker)
enable_project_sanitizers(tree_artifact_checker)

add_executable(tree_factor_check reference/tools/tree_factor_check.cpp)
target_link_libraries(tree_factor_check PRIVATE tree_artifact_checker)
enable_project_warnings(tree_factor_check)
enable_project_sanitizers(tree_factor_check)

add_executable(tree_factor_device src/tree_factor_device.cpp)
target_include_directories(tree_factor_device PRIVATE third_party)
target_link_libraries(tree_factor_device PRIVATE controlplane)
enable_project_warnings(tree_factor_device)
enable_project_sanitizers(tree_factor_device)

add_executable(assembly_real_tests tests/assembly_real_tests.cpp)
target_link_libraries(assembly_real_tests PRIVATE controlplane)
enable_project_warnings(assembly_real_tests)
enable_project_sanitizers(assembly_real_tests)
add_executable(tree_real_tests tests/tree_real_tests.cpp)
target_include_directories(tree_real_tests PRIVATE third_party)
target_link_libraries(tree_real_tests PRIVATE controlplane tree_artifact_checker)
enable_project_warnings(tree_real_tests)
enable_project_sanitizers(tree_real_tests)
add_executable(tree_real_systemc_tests tests/tree_real_systemc_tests.cpp)
target_link_libraries(tree_real_systemc_tests PRIVATE controlplane tree_artifact_checker SystemC::systemc)
enable_project_warnings(tree_real_systemc_tests)
enable_project_sanitizers(tree_real_systemc_tests)
if(SYSTEMC_HOME AND APPLE)
    set_target_properties(tree_real_systemc_tests PROPERTIES BUILD_RPATH "${SYSTEMC_HOME}/lib")
endif()

add_executable(controlplane_core_tests tests/controlplane_core_tests.cpp)
target_include_directories(controlplane_core_tests PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/third_party")
target_link_libraries(controlplane_core_tests PRIVATE controlplane)
enable_project_warnings(controlplane_core_tests)
enable_project_sanitizers(controlplane_core_tests)

add_executable(controlplane_systemc_tests tests/controlplane_systemc_tests.cpp)
target_include_directories(controlplane_systemc_tests PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/third_party")
target_link_libraries(controlplane_systemc_tests PRIVATE controlplane SystemC::systemc)
enable_project_warnings(controlplane_systemc_tests)
enable_project_sanitizers(controlplane_systemc_tests)
if(SYSTEMC_HOME AND APPLE)
    set_target_properties(
        controlplane_systemc_tests PROPERTIES BUILD_RPATH "${SYSTEMC_HOME}/lib")
endif()

add_executable(controlplane_artifact_tests tests/controlplane_artifact_tests.cpp)
target_include_directories(controlplane_artifact_tests PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/third_party")
target_link_libraries(controlplane_artifact_tests PRIVATE controlplane)
enable_project_warnings(controlplane_artifact_tests)
enable_project_sanitizers(controlplane_artifact_tests)

add_executable(command_device_smoke src/command_device_smoke.cpp)
target_link_libraries(command_device_smoke PRIVATE controlplane)
enable_project_warnings(command_device_smoke)
enable_project_sanitizers(command_device_smoke)

# ---------------------------------------------------------------------------
# T06 Panel-LU architecture + multi-mode PE timing/resource prototype
# ---------------------------------------------------------------------------

add_library(panel_lu STATIC
    src/panel_lu/task_types.cpp
    src/panel_lu/tile_kernels.cpp
    src/panel_lu/tile_store.cpp
    src/panel_lu/pe_model.cpp
    src/panel_lu/tile_dispatcher.cpp
    src/panel_lu/panel_controller.cpp
    src/panel_lu/trsm_controller.cpp
    src/panel_lu/gemm_types.cpp
    src/panel_lu/precision_adapter.cpp
    src/panel_lu/gemm_controller.cpp
    src/panel_lu/panel_engine.cpp
    src/panel_lu/panel_lu_prototype.cpp
)
target_include_directories(panel_lu
    PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_link_libraries(panel_lu PUBLIC dataplane)
enable_project_warnings(panel_lu)
enable_project_sanitizers(panel_lu)

add_executable(panel_lu_prototype_tests tests/panel_lu_prototype_tests.cpp)
target_link_libraries(panel_lu_prototype_tests PRIVATE panel_lu)
enable_project_warnings(panel_lu_prototype_tests)
enable_project_sanitizers(panel_lu_prototype_tests)

add_executable(panel_lu_systemc_tests tests/panel_lu_systemc_tests.cpp)
target_link_libraries(panel_lu_systemc_tests PRIVATE panel_lu SystemC::systemc)
enable_project_warnings(panel_lu_systemc_tests)
enable_project_sanitizers(panel_lu_systemc_tests)
if(SYSTEMC_HOME AND APPLE)
    set_target_properties(
        panel_lu_systemc_tests PROPERTIES BUILD_RPATH "${SYSTEMC_HOME}/lib")
endif()

add_executable(panel_lu_real_systemc_tests tests/panel_lu_real_systemc_tests.cpp)
target_link_libraries(panel_lu_real_systemc_tests
    PRIVATE controlplane panel_lu fp64_reference SystemC::systemc)
enable_project_warnings(panel_lu_real_systemc_tests)
enable_project_sanitizers(panel_lu_real_systemc_tests)
if(SYSTEMC_HOME AND APPLE)
    set_target_properties(
        panel_lu_real_systemc_tests PROPERTIES BUILD_RPATH "${SYSTEMC_HOME}/lib")
endif()

add_executable(trsm_real_tests tests/trsm_real_tests.cpp)
target_link_libraries(trsm_real_tests PRIVATE controlplane panel_lu fp64_reference)
enable_project_warnings(trsm_real_tests)
enable_project_sanitizers(trsm_real_tests)

add_executable(gemm_real_tests tests/gemm_real_tests.cpp)
target_include_directories(gemm_real_tests PRIVATE third_party)
target_link_libraries(gemm_real_tests PRIVATE controlplane panel_lu fp64_reference)
enable_project_warnings(gemm_real_tests)
enable_project_sanitizers(gemm_real_tests)

add_executable(panel_lu_real_tests tests/panel_lu_real_tests.cpp)
target_link_libraries(panel_lu_real_tests PRIVATE controlplane panel_lu fp64_reference)
enable_project_warnings(panel_lu_real_tests)
enable_project_sanitizers(panel_lu_real_tests)

add_executable(panel_lu_prototype_smoke src/panel_lu_prototype_smoke.cpp)
target_link_libraries(panel_lu_prototype_smoke PRIVATE panel_lu)
enable_project_warnings(panel_lu_prototype_smoke)
enable_project_sanitizers(panel_lu_prototype_smoke)

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

    add_test(NAME controlplane_core_tests COMMAND controlplane_core_tests)

    add_test(NAME controlplane_systemc_tests COMMAND controlplane_systemc_tests)

    add_test(NAME controlplane_artifact_tests COMMAND controlplane_artifact_tests
        "${T04_FIXTURE_DIR}")
    set_tests_properties(controlplane_artifact_tests PROPERTIES
        DEPENDS t04_generate_fixtures)

    add_test(NAME command_device_smoke COMMAND command_device_smoke
        --device-dir "${T04_FIXTURE_DIR}/multiroot_static/device"
        --out "${CMAKE_CURRENT_BINARY_DIR}/t05_output/smoke")
    set_tests_properties(command_device_smoke PROPERTIES
        DEPENDS t04_generate_fixtures)

    add_test(NAME panel_lu_prototype_tests COMMAND panel_lu_prototype_tests
        "${CMAKE_CURRENT_BINARY_DIR}/t06_output")

    add_test(NAME panel_lu_real_tests COMMAND panel_lu_real_tests
        "${CMAKE_CURRENT_BINARY_DIR}/t07_output")
    add_test(NAME trsm_real_tests COMMAND trsm_real_tests
        "${CMAKE_CURRENT_BINARY_DIR}/t09_output")
    set(T10_FIXTURE_DIR "${CMAKE_CURRENT_BINARY_DIR}/t10_fixtures")
    add_test(
        NAME t10_generate_fixtures
        COMMAND "${T04_PYTHON}"
            "${CMAKE_CURRENT_SOURCE_DIR}/tools/t10_fixture_gen.py"
            --out "${T10_FIXTURE_DIR}")

    add_test(NAME gemm_real_tests COMMAND gemm_real_tests
        "${CMAKE_CURRENT_BINARY_DIR}/t10_output"
        "${T10_FIXTURE_DIR}/int32_minimal/device"
        "${T10_FIXTURE_DIR}/int32_configured/device")
    set_tests_properties(gemm_real_tests PROPERTIES
        DEPENDS t10_generate_fixtures)
    set(T11_FIXTURE_DIR "${CMAKE_CURRENT_BINARY_DIR}/t11_fixtures")
    add_test(NAME t11_generate_fixtures COMMAND "${T04_PYTHON}"
        "${CMAKE_CURRENT_SOURCE_DIR}/tools/t11_fixture_gen.py" --out "${T11_FIXTURE_DIR}")
    add_test(NAME assembly_real_tests COMMAND assembly_real_tests)
    add_test(NAME tree_real_tests COMMAND tree_real_tests "${T11_FIXTURE_DIR}"
        "${CMAKE_CURRENT_BINARY_DIR}/t11_output")
    set_tests_properties(tree_real_tests PROPERTIES DEPENDS t11_generate_fixtures)
    add_test(NAME tree_real_systemc_tests COMMAND tree_real_systemc_tests "${T11_FIXTURE_DIR}"
        "${CMAKE_CURRENT_BINARY_DIR}/t11_systemc_output")
    set_tests_properties(tree_real_systemc_tests PROPERTIES DEPENDS t11_generate_fixtures)
    foreach(mode fp32 int32)
        if(mode STREQUAL "fp32")
            set(T11_BACKEND "FP32_SYSTEMC")
        else()
            set(T11_BACKEND "FP32_SYSTEMC_INT32_GEMM")
        endif()
        add_test(NAME tree_factor_device_${mode} COMMAND tree_factor_device
            --device-dir "${T11_FIXTURE_DIR}/three_level/${mode}/device"
            --out "${CMAKE_CURRENT_BINARY_DIR}/t11_cli/${mode}" --backend "${T11_BACKEND}")
        set_tests_properties(tree_factor_device_${mode} PROPERTIES DEPENDS t11_generate_fixtures)
        add_test(NAME tree_factor_check_${mode} COMMAND tree_factor_check
            "${T11_FIXTURE_DIR}/three_level/${mode}/device" "${CMAKE_CURRENT_BINARY_DIR}/t11_cli/${mode}")
        set_tests_properties(tree_factor_check_${mode} PROPERTIES DEPENDS tree_factor_device_${mode})
    endforeach()
    add_test(NAME tree_factor_device_backend_mismatch COMMAND tree_factor_device
        --device-dir "${T11_FIXTURE_DIR}/three_level/int32/device"
        --out "${CMAKE_CURRENT_BINARY_DIR}/t11_cli/backend_mismatch" --backend FP32_SYSTEMC)
    add_test(NAME tree_factor_device_pivot_failure COMMAND tree_factor_device
        --device-dir "${T11_FIXTURE_DIR}/pivot_propagation/int32/device"
        --out "${CMAKE_CURRENT_BINARY_DIR}/t11_cli/pivot_failure"
        --backend FP32_SYSTEMC_INT32_GEMM)
    add_test(NAME tree_factor_device_timeout COMMAND tree_factor_device
        --device-dir "${T11_FIXTURE_DIR}/three_level/fp32/device"
        --out "${CMAKE_CURRENT_BINARY_DIR}/t11_cli/timeout" --max-cycles 1)
    set_tests_properties(tree_factor_device_backend_mismatch tree_factor_device_pivot_failure
        tree_factor_device_timeout PROPERTIES DEPENDS t11_generate_fixtures WILL_FAIL TRUE)
    add_test(NAME panel_lu_real_systemc_tests COMMAND panel_lu_real_systemc_tests)

    add_test(NAME panel_lu_systemc_tests COMMAND panel_lu_systemc_tests)
    set_tests_properties(panel_lu_systemc_tests PROPERTIES
        DEPENDS panel_lu_prototype_tests)

    # Cross-tile tail trace (p = 23 > 16) and mixed multi-mode load, generated
    # as reproducible T06 evidence.
    add_test(
        NAME panel_lu_smoke_tail
        COMMAND panel_lu_prototype_smoke
            --mode tail --front-dim 23 --pivot-dim 23 --pe-count 1
            --trace-dir "${CMAKE_CURRENT_BINARY_DIR}/t06_output/tail_p23_pe1")

    add_test(
        NAME panel_lu_smoke_mixed
        COMMAND panel_lu_prototype_smoke
            --mode mixed --front-dim 40 --pivot-dim 40 --pe-count 2
            --trace-dir "${CMAKE_CURRENT_BINARY_DIR}/t06_output/mixed_p40_pe2")
endif()
