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
        target_link_options(${target} PRIVATE
            -fsanitize=address,undefined
        )
    endif()
endfunction()

# Prefer the package exported by SystemC 3.x. SYSTEMC_HOME can be supplied either
# as an environment variable or as a CMake cache variable.
set(SYSTEMC_HOME "$ENV{SYSTEMC_HOME}" CACHE PATH "SystemC installation root")
if(SYSTEMC_HOME)
    list(PREPEND CMAKE_PREFIX_PATH "${SYSTEMC_HOME}")
endif()
find_package(SystemCLanguage CONFIG REQUIRED)

function(add_systemc_model target source)
    add_executable(${target} ${source})
    target_include_directories(${target} PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
        "${CMAKE_CURRENT_SOURCE_DIR}/third_party"
    )
    target_link_libraries(${target} PRIVATE SystemC::systemc)
    enable_project_sanitizers(${target})
    target_compile_options(${target} PRIVATE
        $<$<CXX_COMPILER_ID:AppleClang,Clang,GNU>:-Wall;-Wextra;-Wpedantic>
    )
    if(SYSTEMC_HOME AND APPLE)
        set_target_properties(${target} PROPERTIES BUILD_RPATH "${SYSTEMC_HOME}/lib")
    endif()
endfunction()

add_systemc_model(atu_hpu_demo src/atu_hpu_demo.cpp)
add_systemc_model(system_sim src/system_sim.cpp)

add_executable(model_unit_tests src/model_unit_tests.cpp)
target_include_directories(model_unit_tests PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_include_directories(model_unit_tests PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/third_party")
target_compile_options(model_unit_tests PRIVATE
    $<$<CXX_COMPILER_ID:AppleClang,Clang,GNU>:-Wall;-Wextra;-Wpedantic>
)
target_link_libraries(model_unit_tests PRIVATE SystemC::systemc)
enable_project_sanitizers(model_unit_tests)
if(SYSTEMC_HOME AND APPLE)
    set_target_properties(
        model_unit_tests PROPERTIES BUILD_RPATH "${SYSTEMC_HOME}/lib")
endif()

include(CTest)
if(BUILD_TESTING)
    find_package(Python3 COMPONENTS Interpreter REQUIRED)
    add_test(NAME atu_hpu_demo COMMAND atu_hpu_demo)
    add_test(NAME model_unit_tests COMMAND model_unit_tests)
    add_test(
        NAME system_sim_e2e
        COMMAND
            "${Python3_EXECUTABLE}"
            "${CMAKE_CURRENT_SOURCE_DIR}/tests/run_e2e.py"
            "$<TARGET_FILE:system_sim>"
            "${CMAKE_CURRENT_SOURCE_DIR}/../software"
            "${CMAKE_CURRENT_SOURCE_DIR}/config/default.json"
    )
    set_tests_properties(
        system_sim_e2e PROPERTIES ENVIRONMENT "PYTHONDONTWRITEBYTECODE=1")
endif()
