cmake_minimum_required(VERSION 3.16)

project(legacy_graduation_systemc_models LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(SYSTEMC_HOME "$ENV{SYSTEMC_HOME}" CACHE PATH "SystemC installation root")
if(SYSTEMC_HOME)
    list(PREPEND CMAKE_PREFIX_PATH "${SYSTEMC_HOME}")
endif()
find_package(SystemCLanguage CONFIG REQUIRED)

function(add_legacy_systemc_model target source)
    add_executable(${target} ${source})
    target_include_directories(${target} PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
        "${CMAKE_CURRENT_SOURCE_DIR}/third_party")
    target_link_libraries(${target} PRIVATE SystemC::systemc)
    target_compile_options(${target} PRIVATE
        $<$<CXX_COMPILER_ID:AppleClang,Clang,GNU>:-Wall;-Wextra;-Wpedantic>)
    if(SYSTEMC_HOME AND APPLE)
        set_target_properties(${target} PROPERTIES BUILD_RPATH "${SYSTEMC_HOME}/lib")
    endif()
endfunction()

add_legacy_systemc_model(system_sim src/system_sim.cpp)
add_legacy_systemc_model(model_unit_tests src/model_unit_tests.cpp)

include(CTest)
if(BUILD_TESTING)
    find_package(Python3 COMPONENTS Interpreter REQUIRED)
    add_test(NAME model_unit_tests COMMAND model_unit_tests)
    add_test(
        NAME system_sim_e2e
        COMMAND
            "${Python3_EXECUTABLE}"
            "${CMAKE_CURRENT_SOURCE_DIR}/tests/run_e2e.py"
            "$<TARGET_FILE:system_sim>"
            "${CMAKE_CURRENT_SOURCE_DIR}/../software"
            "${CMAKE_CURRENT_SOURCE_DIR}/config/default.json")
endif()
