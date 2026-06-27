On Windows, HLSL compilation fails with:

```txt
Failed to compile C:/Users/johan/Repos/vxray/src/fullscreen.ps.hlsl to SPIRV: HLSL compilation failed: Unknown argument: '-fspv-preserve-bindings'
Failed to compile C:/Users/johan/Repos/vxray/src/fullscreen.vs.hlsl to SPIRV: HLSL compilation failed: Unknown argument: '-fspv-preserve-bindings'
```

Likely cause:

- SDL_shadercross issue: https://github.com/libsdl-org/SDL_shadercross/issues/127
- SDL_shadercross now preserves unused HLSL resource bindings by default so SDL_GPU binding slots and reflection stay stable.
- The fix makes SDL_shadercross pass `-fspv-preserve-bindings` to DXC.
- This error means the `dxcompiler.dll` used at runtime is probably too old and does not recognize that flag.
- Maybe the local `dxcompiler.dll` used at runtime is too old and doesn't recognize the flag?

Preferred fix:

- Ensure the runtime DXC version is new enough for the pinned SDL_shadercross revision (DirectXShaderCompiler: `1f679a48f`)

Temporary workaround:

- Opt back into culling unused bindings when compiling HLSL to SPIR-V. This should suppress `-fspv-preserve-bindings`, but it also opts out of the issue #127 fix. Probably won't use the feature anyway.

In `src/shader_compiler.cc`, create shader properties before `SDL_ShaderCross_HLSL_Info` and pass them through `info.props`:

```cpp
SDL_PropertiesID const shader_props = SDL_CreateProperties();
if (!shader_props)
{
    fprintf(stderr, "Failed to create shader properties: %s\n", SDL_GetError());
    return std::nullopt;
}
scope_guard shader_props_guard{[shader_props] { SDL_DestroyProperties(shader_props); }};

SDL_SetBooleanProperty(
    shader_props,
    SDL_SHADERCROSS_PROP_SHADER_CULL_UNUSED_BINDINGS_BOOLEAN,
    true);

SDL_ShaderCross_HLSL_Info info{
    .source = (char const*)source.data(),
    .entrypoint = "main",
    .include_dir = input_dir_string.c_str(),
    .defines = 0,
    .shader_stage = stage,
    .props = shader_props};
```
