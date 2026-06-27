#include <SDL3/SDL_error.h>
#include <SDL3/SDL_filesystem.h>
#include <SDL3/SDL_iostream.h>
#include <SDL3/SDL_platform_defines.h>
#include <SDL3/SDL_stdinc.h>
#include <SDL3_shadercross/SDL_shadercross.h>

#include <assert.h>
#include <ctype.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_PATH_LENGTH 512
#define MAX_SYMBOL_LENGTH 256
#define MAX_SHADER_LENGTH (1 << 13) // 8 KiB
#define MAX_COMMAND_LENGTH 1024

typedef struct compiled_shader
{
    char    symbol[MAX_SYMBOL_LENGTH];
    uint8_t metal_bytes[MAX_SHADER_LENGTH];
    size_t  metal_size;
    uint8_t spirv_bytes[MAX_SHADER_LENGTH];
    size_t  spirv_size;
} compiled_shader;

static bool has_suffix(char const* const string, char const* const suffix)
{
    size_t const string_length = strlen(string);
    size_t const suffix_length = strlen(suffix);
    if (string_length < suffix_length)
    {
        return false;
    }
    return strcmp(string + string_length - suffix_length, suffix) == 0;
}

static void split_path(
    char const* const path, char* const dir, size_t const dir_size, char* const stem,
    size_t const stem_size)
{
    char const* const slash = strrchr(path, '/');
    char const* const filename = slash ? slash + 1 : path;

    if (slash)
    {
        size_t const length = (size_t)(slash - path);
        size_t const copy_length = length < dir_size - 1 ? length : dir_size - 1;
        memcpy(dir, path, copy_length);
        dir[copy_length] = '\0';
    }
    else
    {
        snprintf(dir, dir_size, ".");
    }

    char const* const dot = strrchr(filename, '.');
    size_t const      filename_length = dot ? (size_t)(dot - filename) : strlen(filename);
    size_t const copy_length = filename_length < stem_size - 1 ? filename_length : stem_size - 1;
    memcpy(stem, filename, copy_length);
    stem[copy_length] = '\0';
}

static void mangle_symbol(char const* const stem, char* const out, size_t const out_size)
{
    snprintf(out, out_size, "%s", stem);
    for (char* c = out; *c; ++c)
    {
        if (*c == '-' || *c == '.')
        {
            *c = '_';
        }
        *c = (char)toupper((unsigned char)*c);
    }
}

static bool run_command(char const* const command)
{
    int const result = system(command);
    return result == 0;
}

static bool compile_shader(char const* const input_file, compiled_shader* const result)
{
    bool success = false;

    if (!(has_suffix(input_file, "vs.hlsl") || has_suffix(input_file, "ps.hlsl") ||
          has_suffix(input_file, "cs.hlsl")))
    {
        fprintf(stderr, "%s has unexpected suffix (expected one of vs, ps, cs)\n", input_file);
        return success;
    }

    char dir[MAX_PATH_LENGTH];
    char stem[MAX_PATH_LENGTH];
    split_path(input_file, dir, sizeof(dir), stem, sizeof(stem));
    mangle_symbol(stem, result->symbol, sizeof(result->symbol));

    size_t      hlsl_size;
    char* const hlsl = SDL_LoadFile(input_file, &hlsl_size);
    if (!hlsl)
    {
        fprintf(stderr, "Failed to load %s\n", input_file);
        return success;
    }
    assert(hlsl_size > 0);
    // TODO: should hlsl be null-terminated? This might be a bug

    SDL_ShaderCross_ShaderStage stage;
    if (has_suffix(input_file, "vs.hlsl"))
    {
        stage = SDL_SHADERCROSS_SHADERSTAGE_VERTEX;
    }
    else if (has_suffix(input_file, "ps.hlsl"))
    {
        stage = SDL_SHADERCROSS_SHADERSTAGE_FRAGMENT;
    }
    else
    {
        stage = SDL_SHADERCROSS_SHADERSTAGE_COMPUTE;
    }

    SDL_ShaderCross_HLSL_Info info = {
        .source = hlsl,
        .entrypoint = "main",
        .include_dir = dir,
        .defines = 0,
        .shader_stage = stage,
        .props = 0};

    size_t         spirv_size = 0;
    uint8_t* const spirv = SDL_ShaderCross_CompileSPIRVFromHLSL(&info, &spirv_size);
    if (!spirv)
    {
        fprintf(stderr, "Failed to compile SPIR-V from HLSL: %s\n", SDL_GetError());
        goto hlsl_cleanup;
    }
    if (spirv_size >= MAX_SHADER_LENGTH)
    {
        fprintf(stderr, "SPIRV is too large, increase MAX_SHADER_LENGTH\n");
        goto spirv_cleanup;
    }

    SDL_ShaderCross_SPIRV_Info const spirv_info = {
        .bytecode = spirv,
        .bytecode_size = spirv_size,
        .entrypoint = "main",
        .shader_stage = stage,
        .props = 0};
    char* const msl = (char*)SDL_ShaderCross_TranspileMSLFromSPIRV(&spirv_info);
    if (!msl)
    {
        fprintf(stderr, "Failed to compile MSL from SPIRV: %s\n", SDL_GetError());
        goto msl_cleanup;
    }

    char metal_path[MAX_PATH_LENGTH];
    char air_path[MAX_PATH_LENGTH];
    char metallib_path[MAX_PATH_LENGTH];
    snprintf(metal_path, sizeof(metal_path), "%s/%s.dda.generated.metal", dir, stem);
    snprintf(air_path, sizeof(air_path), "%s/%s.dda.generated.air", dir, stem);
    snprintf(metallib_path, sizeof(metallib_path), "%s/%s.dda.generated.metallib", dir, stem);

    {
        FILE* const metal_file = fopen(metal_path, "w");
        if (!metal_file)
        {
            fprintf(stderr, "Failed to open %s for writing\n", metal_path);
            goto metal_cleanup;
        }
        fputs(msl, metal_file);
        fclose(metal_file);
    }

#if defined(SDL_PLATFORM_APPLE)
    char const* const metal_command_prefix = "xcrun -sdk macosx";
#else
    char const* const metal_command_prefix = "";
#endif

    char command[MAX_COMMAND_LENGTH];

    snprintf(
        command, sizeof(command), "%s metal -o \"%s\" -c \"%s\"", metal_command_prefix, air_path,
        metal_path);
    if (!run_command(command))
    {
        fprintf(stderr, "Failed to compile generated MSL %s\n", metal_path);
        goto air_cleanup;
    }

    snprintf(
        command, sizeof(command), "%s metallib \"%s\" -o \"%s\"", metal_command_prefix, air_path,
        metallib_path);
    if (!run_command(command))
    {
        fprintf(stderr, "Failed to compile generated AIR %s\n", air_path);
        goto metallib_cleanup;
    }

    size_t   compiled_msl_size;
    uint8_t* compiled_msl = SDL_LoadFile(metallib_path, &compiled_msl_size);
    if (!compiled_msl)
    {
        fprintf(stderr, "Failed to load generated AIR %s\n", metallib_path);
        goto metallib_cleanup;
    }
    if (compiled_msl_size >= MAX_SHADER_LENGTH)
    {
        fprintf(stderr, "Compiled metallib is too large, increase MAX_SHADER_LENGTH\n");
        goto compiled_msl_cleanup;
    }

    assert(compiled_msl_size <= MAX_SHADER_LENGTH);
    memcpy(result->metal_bytes, compiled_msl, compiled_msl_size);
    result->metal_size = compiled_msl_size;
    assert(spirv_size <= MAX_SHADER_LENGTH);
    memcpy(result->spirv_bytes, spirv, spirv_size);
    result->spirv_size = spirv_size;

    success = true;

compiled_msl_cleanup:
    SDL_free(compiled_msl);

metallib_cleanup:
    SDL_RemovePath(metallib_path);

air_cleanup:
    SDL_RemovePath(air_path);

metal_cleanup:
    SDL_RemovePath(metal_path);

msl_cleanup:
    SDL_free(msl);

spirv_cleanup:
    SDL_free(spirv);

hlsl_cleanup:
    SDL_free(hlsl);

    return success;
}

typedef struct shader_output
{
    char const*    symbol;
    uint8_t const* bytes;
    size_t         size;

} shader_output;

typedef struct output_info
{
    char const*          out_header_path;
    char const*          out_source_path;
    shader_output const* shader_results;
    int                  shader_count;

} output_info;

static bool write_output(output_info const* const info)
{
    bool success = false;

    FILE* const h_file = fopen(info->out_header_path, "w");
    if (!h_file)
    {
        fprintf(stderr, "Failed to open %s for writing\n", info->out_header_path);
        return success;
    }

    FILE* const c_file = fopen(info->out_source_path, "w");
    if (!c_file)
    {
        fprintf(stderr, "Failed to open %s for writing\n", info->out_source_path);
        goto header_cleanup;
    }

    fprintf(
        h_file,
        "// This file is generated by shader-compiler. Do not manually edit this file.\n"
        "\n"
        "#pragma once\n"
        "\n"
        "#include <stddef.h>\n"
        "#include <stdint.h>\n");

    char const* const output_h_slash = strrchr(info->out_header_path, '/');
    char const* const output_h_include =
        output_h_slash ? output_h_slash + 1 : info->out_header_path;

    fprintf(
        c_file,
        "// This file is generated by shader-compiler. Do not manually edit this file.\n"
        "\n"
        "#include \"%s\"\n"
        "\n"
        "#include <stddef.h>\n"
        "#include <stdint.h>\n",
        output_h_include);

    for (int i = 0; i < info->shader_count; ++i)
    {
        shader_output const* const shader = &info->shader_results[i];
        fprintf(h_file, "\nextern uint8_t const %s_BYTES[];\n", shader->symbol);
        fprintf(h_file, "extern size_t const %s_SIZE;\n", shader->symbol);

        fprintf(c_file, "\nuint8_t const %s_BYTES[] = {", shader->symbol);
        size_t const array_size = shader->size > 0 ? shader->size : 1;
        for (size_t byte_index = 0; byte_index < array_size; ++byte_index)
        {
            if (byte_index % 16 == 0)
            {
                fprintf(c_file, "\n    ");
            }
            uint8_t const byte = byte_index < shader->size ? shader->bytes[byte_index] : 0;
            fprintf(c_file, "0x%02x", byte);
            if (byte_index + 1 < array_size)
            {
                fprintf(c_file, ", ");
            }
        }
        fprintf(c_file, "\n};\n");
        fprintf(c_file, "size_t const %s_SIZE = %zu;\n", shader->symbol, shader->size);
    }

    success = true;

source_cleanup:
    fclose(c_file);

header_cleanup:
    fclose(h_file);

    return success;
}

int main(int const argc, char** argv)
{
    if (argc < 6)
    {
        fprintf(
            stderr, "Usage: %s <metal.h> <metal.c> <spirv.h> <spirv.c> <shader.hlsl>...\n",
            argv[0]);
        return EXIT_FAILURE;
    }

    if (!SDL_ShaderCross_Init())
    {
        fprintf(stderr, "Failed to initialize SDL_shadercross: %s\n", SDL_GetError());
        return EXIT_FAILURE;
    }

    int result = EXIT_FAILURE;

    int const shader_count = argc - 5;
    assert(shader_count > 0);

    compiled_shader* const compiled_shaders = calloc((size_t)shader_count, sizeof(compiled_shader));
    assert(compiled_shaders);

    for (int i = 0; i < shader_count; ++i)
    {
        char const* const input_file = argv[i + 5];
        if (!compile_shader(input_file, &compiled_shaders[i]))
        {
            goto cleanup_compiled_shaders;
        }
    }

    shader_output* const metal_outputs = calloc((size_t)shader_count, sizeof(shader_output));
    shader_output* const spirv_outputs = calloc((size_t)shader_count, sizeof(shader_output));
    assert(metal_outputs);
    assert(spirv_outputs);
    for (int i = 0; i < shader_count; ++i)
    {
        compiled_shader const* const compiled_shader = &compiled_shaders[i];
        metal_outputs[i] = (shader_output){.symbol = compiled_shader->symbol,
                                           .bytes = compiled_shader->metal_bytes,
                                           .size = compiled_shader->metal_size};
        spirv_outputs[i] = (shader_output){.symbol = compiled_shader->symbol,
                                           .bytes = compiled_shader->spirv_bytes,
                                           .size = compiled_shader->spirv_size};
    }

    char const* const metal_output_h = argv[1];
    char const* const metal_output_c = argv[2];
    char const* const spirv_output_h = argv[3];
    char const* const spirv_output_c = argv[4];

    if (!write_output(&(output_info){.out_header_path = metal_output_h,
                                     .out_source_path = metal_output_c,
                                     .shader_results = metal_outputs,
                                     .shader_count = shader_count}))
    {
        goto cleanup_shader_outputs;
    }

    if (!write_output(&(output_info){.out_header_path = spirv_output_h,
                                     .out_source_path = spirv_output_c,
                                     .shader_results = spirv_outputs,
                                     .shader_count = shader_count}))
    {
        goto cleanup_shader_outputs;
    }

    result = EXIT_SUCCESS;

cleanup_shader_outputs:
    free(metal_outputs);
    free(spirv_outputs);

cleanup_compiled_shaders:
    free(compiled_shaders);

    SDL_ShaderCross_Quit();

    return result;
}
