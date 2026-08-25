cmake_minimum_required(VERSION 3.21)

# Pin the architecture since sandboxed (Codex) CMake regeneration can't use sysctl to
# infer it. This results in a different config and a full SDL_shadercross dependency rebuild.
if(CMAKE_HOST_SYSTEM_NAME STREQUAL "Darwin" AND NOT DEFINED CMAKE_OSX_ARCHITECTURES)
    set(CMAKE_OSX_ARCHITECTURES arm64 CACHE STRING "macOS build architectures")
endif()

if(CMAKE_GENERATOR STREQUAL "Xcode" AND NOT CMAKE_OSX_DEPLOYMENT_TARGET)
    set(CMAKE_OSX_DEPLOYMENT_TARGET 12.0)
endif()

project(vxray C CXX)

option(ENABLE_SHADER_DEBUGGING "Record shader source in compiled shaders" OFF)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_subdirectory(3rdparty EXCLUDE_FROM_ALL)

add_executable(shader-compiler ${CMAKE_CURRENT_SOURCE_DIR}/src/shader_compiler.c)
target_link_libraries(shader-compiler PRIVATE SDL3::SDL3-static SDL3_shadercross::SDL3_shadercross)

set(VXRAY_SHADER_SOURCES
        display.ps.hlsl
        fullscreen.vs.hlsl
        gbuffer.ps.hlsl
        rtao.ps.hlsl)
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
        $<$<BOOL:${ENABLE_SHADER_DEBUGGING}>:--debug-shaders>
        ${VXRAY_SHADER_SOURCES}
        COMMENT "Compiling HLSL shaders..."
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        DEPENDS ${VXRAY_SHADER_SOURCES}
                ${CMAKE_CURRENT_SOURCE_DIR}/src/constants.h
                ${CMAKE_CURRENT_SOURCE_DIR}/src/display.h
                ${CMAKE_CURRENT_SOURCE_DIR}/src/gbuffer.h
                ${CMAKE_CURRENT_SOURCE_DIR}/src/rtao.h
                ${CMAKE_CURRENT_SOURCE_DIR}/src/spatial_hash.hlsli
                ${CMAKE_CURRENT_SOURCE_DIR}/src/shared.hlsli)
add_custom_target(compile-vxray-shaders DEPENDS ${VXRAY_COMPILED_METAL_SHADERS_C} ${VXRAY_COMPILED_SPIRV_SHADERS_C})
add_dependencies(compile-vxray-shaders shader-compiler)

add_library(
        vxray-imgui STATIC
        ${CMAKE_CURRENT_SOURCE_DIR}/src/integrations/imgui_sdl3.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/src/integrations/imgui_sdl3.h
        ${cimgui_SOURCE_DIR}/imgui/backends/imgui_impl_sdl3.cpp
        ${cimgui_SOURCE_DIR}/imgui/backends/imgui_impl_sdlgpu3.cpp)
target_include_directories(
        vxray-imgui
        PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src/integrations
        PRIVATE ${cimgui_SOURCE_DIR}/imgui/backends)
target_link_libraries(vxray-imgui PUBLIC cimgui PRIVATE SDL3::SDL3-static)
target_compile_options(
        vxray-imgui
        PRIVATE $<$<COMPILE_LANG_AND_ID:CXX,Clang,AppleClang>:-fno-exceptions>
                $<$<COMPILE_LANG_AND_ID:CXX,MSVC>:/EHsc->)

set(VXRAY_SOURCES
        $<$<STREQUAL:${CMAKE_SYSTEM_NAME},Darwin>:compiled_metal_shaders.c>
        $<$<STREQUAL:${CMAKE_SYSTEM_NAME},Darwin>:compiled_metal_shaders.h>
        $<$<NOT:$<STREQUAL:${CMAKE_SYSTEM_NAME},Darwin>>:compiled_spirv_shaders.c>
        $<$<NOT:$<STREQUAL:${CMAKE_SYSTEM_NAME},Darwin>>:compiled_spirv_shaders.h>
        constants.h
        integrations/cvox.cc
        integrations/cvox.h
        display.h
        display.ps.hlsl
        fullscreen.vs.hlsl
        gbuffer.h
        gbuffer.ps.hlsl
        hlsl_shim.h
        rtao.h
        rtao.ps.hlsl
        spatial_hash.hlsli
        shared.hlsli
        vxray.c)
list(TRANSFORM VXRAY_SOURCES PREPEND ${CMAKE_CURRENT_SOURCE_DIR}/src/)

add_executable(vxray ${VXRAY_SOURCES})
target_link_libraries(vxray PRIVATE cglm ogt SDL3::SDL3-static vxray-imgui)
target_compile_definitions(
        vxray
        PRIVATE CIMGUI_DEFINE_ENUMS_AND_STRUCTS=1)
target_compile_options(
        vxray
        PRIVATE $<$<COMPILE_LANG_AND_ID:CXX,Clang,AppleClang>:-fno-exceptions>
                $<$<COMPILE_LANG_AND_ID:CXX,MSVC>:/EHsc->)
add_dependencies(vxray compile-vxray-shaders)
