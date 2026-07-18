cmake_minimum_required(VERSION 3.21)

project(vxray C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_subdirectory(3rdparty EXCLUDE_FROM_ALL)

add_executable(shader-compiler ${CMAKE_CURRENT_SOURCE_DIR}/src/shader_compiler.c)
target_link_libraries(shader-compiler PRIVATE SDL3::SDL3-static SDL3_shadercross::SDL3_shadercross)

set(VXRAY_SHADER_SOURCES
        dda.ps.hlsl
        dda.vs.hlsl)
list(TRANSFORM VXRAY_SHADER_SOURCES PREPEND ${CMAKE_CURRENT_SOURCE_DIR}/src/)

set(VXRAY_COMPILED_METAL_SHADERS_H ${CMAKE_CURRENT_SOURCE_DIR}/src/compiled_metal_shaders.h)
set(VXRAY_COMPILED_METAL_SHADERS_C ${CMAKE_CURRENT_SOURCE_DIR}/src/compiled_metal_shaders.c)
set(VXRAY_COMPILED_SPIRV_SHADERS_H ${CMAKE_CURRENT_SOURCE_DIR}/src/compiled_spirv_shaders.h)
set(VXRAY_COMPILED_SPIRV_SHADERS_C ${CMAKE_CURRENT_SOURCE_DIR}/src/compiled_spirv_shaders.c)

add_custom_command(
        OUTPUT ${VXRAY_COMPILED_METAL_SHADERS_H} ${VXRAY_COMPILED_METAL_SHADERS_C}
        ${VXRAY_COMPILED_SPIRV_SHADERS_H} ${VXRAY_COMPILED_SPIRV_SHADERS_C}
        COMMAND $<TARGET_FILE:shader-compiler>
        ${VXRAY_COMPILED_METAL_SHADERS_H} ${VXRAY_COMPILED_METAL_SHADERS_C}
        ${VXRAY_COMPILED_SPIRV_SHADERS_H} ${VXRAY_COMPILED_SPIRV_SHADERS_C}
        ${VXRAY_SHADER_SOURCES}
        COMMENT "Compiling HLSL shaders..."
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        DEPENDS ${VXRAY_SHADER_SOURCES})
add_custom_target(compile-vxray-shaders DEPENDS ${VXRAY_COMPILED_METAL_SHADERS_C} ${VXRAY_COMPILED_SPIRV_SHADERS_C})
add_dependencies(compile-vxray-shaders shader-compiler)

set(VXRAY_SOURCES
        $<$<STREQUAL:${CMAKE_SYSTEM_NAME},Darwin>:compiled_metal_shaders.c>
        $<$<STREQUAL:${CMAKE_SYSTEM_NAME},Darwin>:compiled_metal_shaders.h>
        $<$<NOT:$<STREQUAL:${CMAKE_SYSTEM_NAME},Darwin>>:compiled_spirv_shaders.c>
        $<$<NOT:$<STREQUAL:${CMAKE_SYSTEM_NAME},Darwin>>:compiled_spirv_shaders.h>
        constants.h
        cvox.cc
        cvox.h
        dda.h
        dda.ps.hlsl
        dda.vs.hlsl
        hlsl_shim.h
        vxray.c)
list(TRANSFORM VXRAY_SOURCES PREPEND ${CMAKE_CURRENT_SOURCE_DIR}/src/)

add_executable(vxray ${VXRAY_SOURCES})
target_link_libraries(vxray PRIVATE ogt SDL3::SDL3-static)
target_compile_options(
        vxray
        PRIVATE $<$<COMPILE_LANG_AND_ID:CXX,Clang,AppleClang>:-fno-exceptions>
                $<$<COMPILE_LANG_AND_ID:CXX,MSVC>:/EHsc->)
add_dependencies(vxray compile-vxray-shaders)
