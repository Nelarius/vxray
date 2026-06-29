include(FetchContent)

# Declare

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
                     GIT_SHALLOW OFF)

# Fetch

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
