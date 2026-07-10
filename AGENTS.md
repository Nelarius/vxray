# AGENTS.md

## Build System

Vxray uses CMake with multiple build configurations. Common commands:

```sh
# Configure with ccache
cmake -B build -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache -DCMAKE_BUILD_TYPE=Debug`

# Regular build
cmake --build build --target vxray

# Recompile generated shaders only
cmake --build build --target compile-vxray-shaders

# Build the shader compiler
cmake --build build --target shader-compiler

# For agents: skip ccache to avoid sandboxing issues
CCACHE_DISABLE=1 cmake --build build --target vxray
```

## Shader Pipeline

HLSL is compiled into SPIRV bytecode and compiled Metal shaders at build time and embedded as byte arrays:

- Source: `src/*.hlsl`
- Compiled and embedded by the `shader-compiler` tool, invoked via the `compile-vxray-shaders` target
- Output: `src/compiled_*_shaders.c/h`

If the shader compiler changed but the shader source did not, force shader recompilation with:

```sh
touch src/fullscreen.ps.hlsl
cmake --build build --target compile-vxray-shaders
```

### Shader resource bindings

SDL GPU requires specific ordering of the resource bindings.

For vertex shaders:

(t[n], space0): Sampled textures, followed by storage textures, followed by storage buffers
(s[n], space0): Samplers with indices corresponding to the sampled textures
(b[n], space1): Uniform buffers

For pixel shaders:

(t[n], space2): Sampled textures, followed by storage textures, followed by storage buffers
(s[n], space2): Samplers with indices corresponding to the sampled textures
(b[n], space3): Uniform buffers

## Lint command

When done with code changes, the code must be formatted with clang-format:

```bash
clang-format -i modules/polycube/src/allocator.c
```

Both C and HLSL should be formatted with clang-format.

## Code Style

**const-correctness**

Use `const` whenever possible. `const` should be used in function parameters, variables, and data behind a pointer.
`const` should be written on the right side of the type it is annotating. For example,

```c
// a const pointer to const int
int const* const voxel_data;
```

**Variable declarations**

Declare variables at first use and initialize them immediately. Use designated initializers for structs:

```c
SDL_GPUColorTargetInfo const color_target_info = {
        .texture = swapchain_texture,
        .clear_color = (SDL_FColor){0.f, 0.f, 0.f, 0.f},
        .load_op = SDL_GPU_LOADOP_CLEAR,
        .store_op = SDL_GPU_STOREOP_STORE};
```

Use `= {0}` for zero-initialization of structs and arrays.

**Idioms**

- Early return on error with explicit error codes
- Use `0` instead of `NULL` for null pointers
- Explicit casts when converting between types: `(size_t)shader_count`
- Prefix increment/decrement: `++i` not `i++`
