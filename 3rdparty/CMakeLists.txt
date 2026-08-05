include(FetchContent)

# Declare

find_package(Git REQUIRED)

FetchContent_Declare(cglm
                     GIT_REPOSITORY "https://github.com/recp/cglm"
                     GIT_TAG 58d8c15a124c202188d0094705f778f6129a2612 # Jul 29, 2026
                     GIT_SHALLOW ON)

FetchContent_Declare(cimgui
                     GIT_REPOSITORY "https://github.com/cimgui/cimgui"
                     GIT_TAG d298666861ebf00dcfeb2407409931c04e47e33c # May 30, 2026
                     GIT_SHALLOW OFF)

FetchContent_Declare(ogt
                     GIT_REPOSITORY "https://github.com/jpaver/opengametools"
                     GIT_TAG dcabf82345979a2d937a5ea29a0b14683a3aa838 # Mar 20, 2026
                     GIT_SHALLOW ON)

FetchContent_Declare(sdl3
                     GIT_REPOSITORY "https://github.com/libsdl-org/SDL"
                     GIT_TAG 8e37db5e797b6167f3a00d697d816a684bd259c7 # v3.4.10
                     GIT_SHALLOW OFF)

FetchContent_Declare(sdl-shadercross
                     GIT_REPOSITORY "https://github.com/libsdl-org/SDL_shadercross"
                     GIT_TAG 9a461644308366b50f0c364069d762be8e3b3443 # Jun 15, 2026
                     GIT_SHALLOW OFF
                     PATCH_COMMAND
                         ${CMAKE_COMMAND}
                         "-DGIT_EXECUTABLE=${GIT_EXECUTABLE}"
                         "-DPATCH_FILE=${CMAKE_CURRENT_SOURCE_DIR}/patches/sdl-shadercross-preserve-deployment-target.patch"
                         "-DSOURCE_DIR=<SOURCE_DIR>"
                         -P
                         "${CMAKE_CURRENT_SOURCE_DIR}/cmake/apply-patch.cmake")

# Fetch

set(CGLM_STATIC ON CACHE BOOL "" FORCE)
set(CGLM_USE_ANONYMOUS_STRUCT ON CACHE BOOL "" FORCE)
message(STATUS "Fetching cglm...")
FetchContent_MakeAvailable(cglm)

set(IMGUI_STATIC ON CACHE STRING "" FORCE)
message(STATUS "Fetching cimgui...")
FetchContent_MakeAvailable(cimgui)

message(STATUS "Fetching opengametools...")
FetchContent_MakeAvailable(ogt)

set(SDL_STATIC ON CACHE BOOL "" FORCE)
message(STATUS "Fetching SDL...")
FetchContent_MakeAvailable(sdl3)

set(SDLSHADERCROSS_SHARED OFF CACHE BOOL "" FORCE)
set(SDLSHADERCROSS_STATIC ON CACHE BOOL "" FORCE)
set(SDLSHADERCROSS_SPIRVCROSS_SHARED OFF CACHE BOOL "" FORCE)
set(SDLSHADERCROSS_VENDORED ON CACHE BOOL "" FORCE)
set(SDLSHADERCROSS_CLI OFF CACHE BOOL "" FORCE)
message(STATUS "Fetching SDL_shadercross...")
FetchContent_MakeAvailable(sdl-shadercross)

# Configure

add_library(ogt INTERFACE)
target_include_directories(ogt INTERFACE ${ogt_SOURCE_DIR}/src)
