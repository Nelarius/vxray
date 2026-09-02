#include "constants.h"
#include "cvox.h"
#include "display.h"
#include "gbuffer.h"
#include "hlsl_shim.h"
#include "rtao.h"
#include "sky.h"

#include <cglm/struct.h>
#include <cglm/struct/clipspace/persp_lh_zo.h>
#include <cglm/struct/clipspace/view_lh_zo.h>
#include <cglm/util.h>
#include <cimgui.h>
#include <imgui_sdl3.h>

#define SDL_MAIN_USE_CALLBACKS 1
#include <SDL3/SDL_error.h>
#include <SDL3/SDL_events.h>
#include <SDL3/SDL_filesystem.h>
#include <SDL3/SDL_gpu.h>
#include <SDL3/SDL_init.h>
#include <SDL3/SDL_iostream.h>
#include <SDL3/SDL_keyboard.h>
#include <SDL3/SDL_log.h>
#include <SDL3/SDL_main.h>
#include <SDL3/SDL_mouse.h>
#include <SDL3/SDL_pixels.h>
#include <SDL3/SDL_platform_defines.h>
#include <SDL3/SDL_properties.h>
#include <SDL3/SDL_scancode.h>
#include <SDL3/SDL_stdinc.h>
#include <SDL3/SDL_video.h>

#include <assert.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

static_assert(sizeof(float4x4) == 64, "float4x4 must match an HLSL column-major matrix");
static_assert(sizeof(display_uniforms) == 32, "display uniform layout must match HLSL");
static_assert(sizeof(gbuffer_uniforms) == 160, "G-buffer uniform layout must match HLSL");
static_assert(sizeof(rtao_uniforms) == 192, "RTAO uniform layout must match HLSL");
static_assert(sizeof(sky_view_uniforms) == 48, "sky-view uniform layout must match HLSL");
static_assert((VX_AO_HASH_TOUCH_PERIOD & (VX_AO_HASH_TOUCH_PERIOD - 1u)) == 0u,
              "AO hash touch period must be a power of two");
static_assert(VX_AO_HASH_TOUCH_PERIOD <= VX_AO_MAX_CELL_AGE,
              "AO hash entries must be touched before they expire");

typedef struct vx_aadf_uniforms
{
    uint32_t grid_ext;
    uint32_t axis;
    uint32_t initialize;
    uint32_t padding;
} vx_aadf_uniforms;

static_assert(sizeof(vx_aadf_uniforms) == 16, "AADF uniform layout must match HLSL");

#ifdef NDEBUG
#define GPU_DEVICE_DEBUG_MODE false
#else
#define GPU_DEVICE_DEBUG_MODE true
#endif

#if defined(SDL_PLATFORM_APPLE)

#include "compiled_metal_shaders.h"

#define GPU_SHADER_FORMAT SDL_GPU_SHADERFORMAT_METALLIB
#define GPU_SHADER_ENTRYPOINT "main0"
#define GPU_DRIVER_NAME "metal"

#elif defined(SDL_PLATFORM_WINDOWS)

#include "compiled_spirv_shaders.h"

#define GPU_SHADER_FORMAT SDL_GPU_SHADERFORMAT_SPIRV
#define GPU_SHADER_ENTRYPOINT "main"
#define GPU_DRIVER_NAME "vulkan"

#else

#error "Only Apple and Windows platforms are supported!"

#endif

#define vx_buffer_decl(T)                                                                          \
    typedef struct vx_##T##_buffer                                                                 \
    {                                                                                              \
        T*  ptr;                                                                                   \
        int count;                                                                                 \
    } vx_##T##_buffer

#define vx_buffer(T) vx_##T##_buffer

#define vx_buffer_calloc(T, N)                                                                     \
    (vx_##T##_buffer) { .ptr = calloc((N), sizeof(T)), .count = (N) }

#define vx_buffer_free(b) free((b).ptr)

vx_buffer_decl(uint8_t);

static uint32_t vx_next_power_of_2(uint32_t x)
{
    // NOTE: the method returns 0 for x = 0, which isn't a power of 2.
    assert(x > 0);

    --x;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    ++x;

    return x;
}

static mat4s vx_model_transform(cvox_transform const* const transform)
{
    float const components[16] = {
        transform->m00, transform->m01, transform->m02, transform->m03,
        transform->m10, transform->m11, transform->m12, transform->m13,
        transform->m20, transform->m21, transform->m22, transform->m23,
        transform->m30, transform->m31, transform->m32, transform->m33,
    };
    mat4s const magicavoxel_transform = glms_mat4_make(components);

    // MagicaVoxel is Z-up. Swap Y and Z to make the scene Y-up.
    float const axis_swap_components[16] = {
        1.f, 0.f, 0.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 0.f, 0.f, 1.f,
    };
    mat4s const axis_swap = glms_mat4_make(axis_swap_components);
    return glms_mat4_mul(axis_swap, magicavoxel_transform);
}

static float4 vx_float4_from_vec3(vec3s const v, float const w) { return float4(v.x, v.y, v.z, w); }

static float4x4 vx_float4x4_from_mat4(mat4s const matrix)
{
    float4x4 result;
    SDL_memcpy(result.data, matrix.raw, sizeof(result.data));
    return result;
}

static int vx_round_to_int(float const val)
{
    return (int)(val >= 0.f ? (val + 0.5f) : (val - 0.5f));
}

static int vx_grid_index(int16_t const x, int16_t const y, int16_t const z, int const grid_ext)
{
    return x + (y * grid_ext) + (z * grid_ext * grid_ext);
}

// Builds 2x2x2 occupancy masks for a dense grid. Each mask texel contains 8 bits in R8_UINT, with
// x varying fastest, then y, then z.
static bool vx_create_occupancy_masks(vx_buffer(uint8_t) const occupancy_grid,
                                      int const                  occupancy_grid_ext,
                                      vx_buffer(uint8_t) * const out_masks)
{
    assert(occupancy_grid.ptr);
    assert(occupancy_grid_ext > 0);
    assert(out_masks);

    int const mask_grid_ext = (occupancy_grid_ext + VX_MASK_EXT - 1) / VX_MASK_EXT;
    int const mask_count = mask_grid_ext * mask_grid_ext * mask_grid_ext;
    vx_buffer(uint8_t) masks = vx_buffer_calloc(uint8_t, mask_count);
    if (!masks.ptr)
    {
        SDL_LogError(SDL_LOG_CATEGORY_APPLICATION, "Failed to allocate occupancy masks");
        return false;
    }

    for (int16_t z = 0; z < occupancy_grid_ext; ++z)
    {
        for (int16_t y = 0; y < occupancy_grid_ext; ++y)
        {
            for (int16_t x = 0; x < occupancy_grid_ext; ++x)
            {
                if (occupancy_grid.ptr[vx_grid_index(x, y, z, occupancy_grid_ext)] == 0u)
                {
                    continue;
                }

                int const mask_index =
                    vx_grid_index(x / VX_MASK_EXT, y / VX_MASK_EXT, z / VX_MASK_EXT, mask_grid_ext);
                int const bit_index = (x & 1) | ((y & 1) << 1) | ((z & 1) << 2);
                masks.ptr[mask_index] |= (uint8_t)(1u << bit_index);
            }
        }
    }

    *out_masks = masks;
    return true;
}

static bool vx_create_coarse_occupancy_grid(vx_buffer(uint8_t) const occupancy_grid,
                                            int const occupancy_grid_ext, int const cell_ext,
                                            vx_buffer(uint8_t) * const out_grid)
{
    assert(occupancy_grid.ptr);
    assert(occupancy_grid_ext > 0);
    assert(cell_ext > 0);
    assert(out_grid);

    int const coarse_grid_ext = (occupancy_grid_ext + cell_ext - 1) / cell_ext;
    int const coarse_count = coarse_grid_ext * coarse_grid_ext * coarse_grid_ext;
    vx_buffer(uint8_t) coarse_grid = vx_buffer_calloc(uint8_t, coarse_count);
    if (!coarse_grid.ptr)
    {
        SDL_LogError(SDL_LOG_CATEGORY_APPLICATION, "Failed to allocate coarse occupancy grid");
        return false;
    }

    for (int16_t z = 0; z < occupancy_grid_ext; ++z)
    {
        for (int16_t y = 0; y < occupancy_grid_ext; ++y)
        {
            for (int16_t x = 0; x < occupancy_grid_ext; ++x)
            {
                if (occupancy_grid.ptr[vx_grid_index(x, y, z, occupancy_grid_ext)] != 0u)
                {
                    int const index =
                        vx_grid_index(x / cell_ext, y / cell_ext, z / cell_ext, coarse_grid_ext);
                    coarse_grid.ptr[index] = 1u;
                }
            }
        }
    }

    *out_grid = coarse_grid;
    return true;
}

typedef struct vx_scene
{
    vx_buffer(uint8_t) voxel_grid;
    vx_buffer(uint8_t) brick_grid;
    vx_buffer(uint8_t) chunk_grid;
    vx_buffer(uint8_t) voxel_masks;
    vx_buffer(uint8_t) brick_masks;
    vx_buffer(uint8_t) chunk_masks;
    uint  palette[256];
    int   grid_ext;
    int   brick_grid_ext;
    int   chunk_grid_ext;
    vec3s center;
} vx_scene;

// Loads a MagicaVoxel scene into dense voxel, brick, and chunk grids and occupancy masks.
static bool vx_load_scene(char const* const vox_path, vx_scene* const out_scene)
{
    assert(vox_path);
    assert(out_scene);

    vx_buffer(uint8_t) voxel_grid = {0};
    vx_buffer(uint8_t) brick_grid = {0};
    vx_buffer(uint8_t) chunk_grid = {0};
    vx_buffer(uint8_t) voxel_masks = {0};
    vx_buffer(uint8_t) brick_masks = {0};
    vx_buffer(uint8_t) chunk_masks = {0};
    cvox_scene const* scene = 0;
    {
        size_t   num_bytes;
        uint8_t* buffer = SDL_LoadFile(vox_path, &num_bytes);
        if (!buffer || !num_bytes)
        {
            SDL_LogError(SDL_LOG_CATEGORY_APPLICATION, "Failed to load %s", vox_path);
            SDL_free(buffer);
            return false;
        }

        scene = cvox_read_scene(buffer, num_bytes);
        SDL_free(buffer);
    }

    if (!scene)
    {
        SDL_LogError(SDL_LOG_CATEGORY_APPLICATION, "Failed to load scene from %s", vox_path);
        return false;
    }
    if (scene->num_instances == 0 || scene->num_models == 0)
    {
        SDL_LogError(SDL_LOG_CATEGORY_APPLICATION, "Scene has no instances or models");
        goto cleanup_scene;
    }

    {
        int3 scene_min = {.x = INT32_MAX, .y = INT32_MAX, .z = INT32_MAX};
        int3 scene_max = {.x = INT32_MIN, .y = INT32_MIN, .z = INT32_MIN};

        for (int i = 0; i < scene->num_instances; ++i)
        {
            cvox_instance const* const instance = &scene->instances[i];
            int const                  model_idx = (int)cvox_sample_instance_model(instance, 0);
            assert(model_idx < scene->num_models);

            cvox_model const* const model = scene->models[model_idx];
            assert(model);
            if (model->size_x == 0 || model->size_y == 0 || model->size_z == 0)
            {
                // NOTE: empty model -- does model->voxel_hash have a sentinel value we could look
                // up?
                continue;
            }

            cvox_transform const source_transform =
                cvox_sample_instance_transform_global(instance, 0, scene);
            mat4s const transform = vx_model_transform(&source_transform);
            int3 const  pivot = {.x = (int)(model->size_x / 2),
                                 .y = (int)(model->size_y / 2),
                                 .z = (int)(model->size_z / 2)};
            float const min_x = (float)-pivot.x;
            float const min_y = (float)-pivot.y;
            float const min_z = (float)-pivot.z;
            float const max_x = (float)((int)model->size_x - 1 - pivot.x);
            float const max_y = (float)((int)model->size_y - 1 - pivot.y);
            float const max_z = (float)((int)model->size_z - 1 - pivot.z);
            vec3s const corners[8] = {{min_x, min_y, min_z}, {max_x, min_y, min_z},
                                      {min_x, max_y, min_z}, {min_x, min_y, max_z},
                                      {max_x, max_y, min_z}, {max_x, min_y, max_z},
                                      {min_x, max_y, max_z}, {max_x, max_y, max_z}};
            for (int c = 0; c < 8; ++c)
            {
                vec3s const tc = glms_mat4_mulv3(transform, corners[c], 1.f);
                int const   rx = vx_round_to_int(tc.x);
                int const   ry = vx_round_to_int(tc.y);
                int const   rz = vx_round_to_int(tc.z);
                scene_min = (int3){SDL_min(rx, scene_min.x), SDL_min(ry, scene_min.y),
                                   SDL_min(rz, scene_min.z)};
                scene_max = (int3){SDL_max(rx, scene_max.x), SDL_max(ry, scene_max.y),
                                   SDL_max(rz, scene_max.z)};
            }
        }

        if (scene_min.x > scene_max.x || scene_min.y > scene_max.y || scene_min.z > scene_max.z)
        {
            SDL_LogError(SDL_LOG_CATEGORY_APPLICATION,
                         "Scene has no valid instances with non-empty models");
            goto cleanup_scene;
        }

        int const scene_ext_x = scene_max.x - scene_min.x + 1;
        int const scene_ext_y = scene_max.y - scene_min.y + 1;
        int const scene_ext_z = scene_max.z - scene_min.z + 1;
        assert(scene_ext_x > 0);
        assert(scene_ext_y > 0);
        assert(scene_ext_z > 0);
        int const largest_extent = SDL_max(scene_ext_x, SDL_max(scene_ext_y, scene_ext_z));
        assert(largest_extent <= 1024);
        if (largest_extent > 1024)
        {
            SDL_LogError(SDL_LOG_CATEGORY_APPLICATION, "Scene extent is too large: %d",
                         largest_extent);
            goto cleanup_scene;
        }

        int const padded_extent = SDL_max(largest_extent, VX_BRICK_EXT);
        int const grid_ext = (int)vx_next_power_of_2((uint32_t)padded_extent);
        assert(grid_ext % VX_BRICK_EXT == 0);
        int const total_voxels = grid_ext * grid_ext * grid_ext;
        assert(total_voxels % 4 == 0);
        voxel_grid = vx_buffer_calloc(uint8_t, total_voxels);
        if (!voxel_grid.ptr)
        {
            SDL_LogError(SDL_LOG_CATEGORY_APPLICATION, "Failed to allocate voxel grid");
            goto cleanup_scene;
        }

        int const brick_grid_ext = grid_ext / VX_BRICK_EXT;
        int const total_bricks = brick_grid_ext * brick_grid_ext * brick_grid_ext;
        brick_grid = vx_buffer_calloc(uint8_t, total_bricks);
        if (!brick_grid.ptr)
        {
            SDL_LogError(SDL_LOG_CATEGORY_APPLICATION, "Failed to allocate brick grid");
            goto cleanup_grids;
        }

        for (int i = 0; i < (int)scene->num_instances; ++i)
        {
            cvox_instance const* const instance = &scene->instances[i];
            int const                  model_idx = (int)cvox_sample_instance_model(instance, 0);

            cvox_model const* const model = scene->models[model_idx];
            assert(model);

            cvox_transform const source_transform =
                cvox_sample_instance_transform_global(instance, 0, scene);
            mat4s const transform = vx_model_transform(&source_transform);

            int const  sx = (int)model->size_x;
            int const  sy = (int)model->size_y;
            int const  sz = (int)model->size_z;
            int3 const pivot = {.x = (int)(model->size_x / 2),
                                .y = (int)(model->size_y / 2),
                                .z = (int)(model->size_z / 2)};
            for (int16_t z = 0; z < sz; ++z)
            {
                for (int16_t y = 0; y < sy; ++y)
                {
                    for (int16_t x = 0; x < sx; ++x)
                    {
                        int const     src_idx = x + y * sx + z * sx * sy;
                        uint8_t const voxel = model->voxel_data[src_idx];
                        if (voxel)
                        {
                            vec3s const local_coord = {.x = (float)(x - pivot.x),
                                                       .y = (float)(y - pivot.y),
                                                       .z = (float)(z - pivot.z)};
                            vec3s const global_coord = glms_mat4_mulv3(transform, local_coord, 1.f);

                            int const gx = vx_round_to_int(global_coord.x);
                            int const gy = vx_round_to_int(global_coord.y);
                            int const gz = vx_round_to_int(global_coord.z);
                            int const raw_dx = gx - scene_min.x;
                            int const raw_dy = gy - scene_min.y;
                            int const raw_dz = gz - scene_min.z;
                            assert(raw_dx >= 0 && raw_dx < grid_ext);
                            assert(raw_dy >= 0 && raw_dy < grid_ext);
                            assert(raw_dz >= 0 && raw_dz < grid_ext);
                            int16_t const dx = (int16_t)raw_dx;
                            int16_t const dy = (int16_t)raw_dy;
                            int16_t const dz = (int16_t)raw_dz;

                            int const dest_idx = dx + dy * grid_ext + dz * grid_ext * grid_ext;
                            assert(dest_idx >= 0 && dest_idx < voxel_grid.count);
                            voxel_grid.ptr[dest_idx] = voxel;

                            int16_t const brick_x = (int16_t)(dx / VX_BRICK_EXT);
                            int16_t const brick_y = (int16_t)(dy / VX_BRICK_EXT);
                            int16_t const brick_z = (int16_t)(dz / VX_BRICK_EXT);
                            int const     brick_idx =
                                vx_grid_index(brick_x, brick_y, brick_z, brick_grid_ext);
                            assert(brick_idx >= 0 && brick_idx < brick_grid.count);
                            brick_grid.ptr[brick_idx] = 1;
                        }
                    }
                }
            }
        }

        int const chunk_grid_ext = (brick_grid_ext + VX_CHUNK_EXT - 1) / VX_CHUNK_EXT;
        if (!vx_create_coarse_occupancy_grid(brick_grid, brick_grid_ext, VX_CHUNK_EXT,
                                             &chunk_grid) ||
            !vx_create_occupancy_masks(voxel_grid, grid_ext, &voxel_masks) ||
            !vx_create_occupancy_masks(brick_grid, brick_grid_ext, &brick_masks) ||
            !vx_create_occupancy_masks(chunk_grid, chunk_grid_ext, &chunk_masks))
        {
            goto cleanup_grids;
        }

        out_scene->grid_ext = grid_ext;
        out_scene->brick_grid_ext = brick_grid_ext;
        out_scene->chunk_grid_ext = chunk_grid_ext;
        out_scene->center = (vec3s){0.5f * (float)scene_ext_x, 0.5f * (float)scene_ext_y,
                                    0.5f * (float)scene_ext_z};
    }

    out_scene->voxel_grid = voxel_grid;
    out_scene->brick_grid = brick_grid;
    out_scene->chunk_grid = chunk_grid;
    out_scene->voxel_masks = voxel_masks;
    out_scene->brick_masks = brick_masks;
    out_scene->chunk_masks = chunk_masks;
    for (int i = 0; i < 256; ++i)
    {
        cvox_rgba const color = scene->palette.color[i];
        out_scene->palette[i] =
            (uint)color.r | ((uint)color.g << 8u) | ((uint)color.b << 16u) | ((uint)color.a << 24u);
    }
    cvox_destroy_scene(scene);
    return true;

cleanup_grids:
    vx_buffer_free(chunk_masks);
    vx_buffer_free(brick_masks);
    vx_buffer_free(voxel_masks);
    vx_buffer_free(chunk_grid);
    vx_buffer_free(brick_grid);
    vx_buffer_free(voxel_grid);
cleanup_scene:
    assert(scene);
    cvox_destroy_scene(scene);
    return false;
}

static void vx_scene_free(vx_scene* const scene)
{
    assert(scene);
    vx_buffer_free(scene->chunk_grid);
    vx_buffer_free(scene->brick_grid);
    vx_buffer_free(scene->voxel_grid);
    vx_buffer_free(scene->chunk_masks);
    vx_buffer_free(scene->brick_masks);
    vx_buffer_free(scene->voxel_masks);
    scene->chunk_grid = (vx_buffer(uint8_t)){0};
    scene->brick_grid = (vx_buffer(uint8_t)){0};
    scene->voxel_grid = (vx_buffer(uint8_t)){0};
    scene->chunk_masks = (vx_buffer(uint8_t)){0};
    scene->brick_masks = (vx_buffer(uint8_t)){0};
    scene->voxel_masks = (vx_buffer(uint8_t)){0};
}

// Resource creation wrappers that attach a debug name, displayed in graphics debuggers, via the
// properties API.

static SDL_GPUTexture* vx_create_gpu_texture(SDL_GPUDevice* const           device,
                                             SDL_GPUTextureCreateInfo const info,
                                             char const* const              name)
{
    SDL_PropertiesID const props = SDL_CreateProperties();
    SDL_SetStringProperty(props, SDL_PROP_GPU_TEXTURE_CREATE_NAME_STRING, name);
    SDL_GPUTextureCreateInfo named_info = info;
    named_info.props = props;
    SDL_GPUTexture* const texture = SDL_CreateGPUTexture(device, &named_info);
    SDL_DestroyProperties(props);
    return texture;
}

static SDL_GPUBuffer* vx_create_gpu_buffer(SDL_GPUDevice* const          device,
                                           SDL_GPUBufferCreateInfo const info,
                                           char const* const             name)
{
    SDL_PropertiesID const props = SDL_CreateProperties();
    SDL_SetStringProperty(props, SDL_PROP_GPU_BUFFER_CREATE_NAME_STRING, name);
    SDL_GPUBufferCreateInfo named_info = info;
    named_info.props = props;
    SDL_GPUBuffer* const buffer = SDL_CreateGPUBuffer(device, &named_info);
    SDL_DestroyProperties(props);
    return buffer;
}

static SDL_GPUTransferBuffer*
vx_create_gpu_transfer_buffer(SDL_GPUDevice* const                  device,
                              SDL_GPUTransferBufferCreateInfo const info, char const* const name)
{
    SDL_PropertiesID const props = SDL_CreateProperties();
    SDL_SetStringProperty(props, SDL_PROP_GPU_TRANSFERBUFFER_CREATE_NAME_STRING, name);
    SDL_GPUTransferBufferCreateInfo named_info = info;
    named_info.props = props;
    SDL_GPUTransferBuffer* const buffer = SDL_CreateGPUTransferBuffer(device, &named_info);
    SDL_DestroyProperties(props);
    return buffer;
}

static SDL_GPUGraphicsPipeline*
vx_create_gpu_graphics_pipeline(SDL_GPUDevice* const                    device,
                                SDL_GPUGraphicsPipelineCreateInfo const info,
                                char const* const                       name)
{
    SDL_PropertiesID const props = SDL_CreateProperties();
    SDL_SetStringProperty(props, SDL_PROP_GPU_GRAPHICSPIPELINE_CREATE_NAME_STRING, name);
    SDL_GPUGraphicsPipelineCreateInfo named_info = info;
    named_info.props = props;
    SDL_GPUGraphicsPipeline* const pipeline = SDL_CreateGPUGraphicsPipeline(device, &named_info);
    SDL_DestroyProperties(props);
    return pipeline;
}

static SDL_GPUComputePipeline*
vx_create_gpu_compute_pipeline(SDL_GPUDevice* const                   device,
                               SDL_GPUComputePipelineCreateInfo const info, char const* const name)
{
    SDL_PropertiesID const props = SDL_CreateProperties();
    SDL_SetStringProperty(props, SDL_PROP_GPU_COMPUTEPIPELINE_CREATE_NAME_STRING, name);
    SDL_GPUComputePipelineCreateInfo named_info = info;
    named_info.props = props;
    SDL_GPUComputePipeline* const pipeline = SDL_CreateGPUComputePipeline(device, &named_info);
    SDL_DestroyProperties(props);
    return pipeline;
}

static bool vx_gpu_buffer_upload(SDL_GPUDevice* const device, SDL_GPUBuffer* const buffer,
                                 void const* const data, uint32_t const size)
{
    SDL_GPUTransferBuffer* const transfer = vx_create_gpu_transfer_buffer(
        device,
        (SDL_GPUTransferBufferCreateInfo){.usage = SDL_GPU_TRANSFERBUFFERUSAGE_UPLOAD,
                                          .size = size},
        "buffer-upload");
    if (!transfer)
    {
        SDL_LogError(SDL_LOG_CATEGORY_GPU, "Failed to create GPU transfer buffer: %s",
                     SDL_GetError());
        return false;
    }

    void* const mapped = SDL_MapGPUTransferBuffer(device, transfer, false);
    if (!mapped)
    {
        SDL_LogError(SDL_LOG_CATEGORY_GPU, "Failed to map GPU transfer buffer: %s", SDL_GetError());
        SDL_ReleaseGPUTransferBuffer(device, transfer);
        return false;
    }
    SDL_memcpy(mapped, data, size);
    SDL_UnmapGPUTransferBuffer(device, transfer);

    SDL_GPUCommandBuffer* const cmd = SDL_AcquireGPUCommandBuffer(device);
    if (!cmd)
    {
        SDL_LogError(SDL_LOG_CATEGORY_GPU, "Failed to acquire GPU command buffer: %s",
                     SDL_GetError());
        SDL_ReleaseGPUTransferBuffer(device, transfer);
        return false;
    }

    SDL_GPUCopyPass* const copy_pass = SDL_BeginGPUCopyPass(cmd);
    assert(copy_pass);
    SDL_UploadToGPUBuffer(
        copy_pass, &(SDL_GPUTransferBufferLocation){.transfer_buffer = transfer, .offset = 0},
        &(SDL_GPUBufferRegion){.buffer = buffer, .offset = 0, .size = size}, false);
    SDL_EndGPUCopyPass(copy_pass);

    bool const submitted = SDL_SubmitGPUCommandBuffer(cmd);
    SDL_ReleaseGPUTransferBuffer(device, transfer);
    if (!submitted)
    {
        SDL_LogError(SDL_LOG_CATEGORY_GPU, "Failed to submit GPU upload command buffer: %s",
                     SDL_GetError());
        return false;
    }

    return true;
}

static bool vx_gpu_texture_upload(SDL_GPUDevice* const device, SDL_GPUTexture* const texture,
                                  void const* const data, uint32_t const size, uint32_t grid_ext)
{
    SDL_GPUTransferBuffer* const transfer = vx_create_gpu_transfer_buffer(
        device,
        (SDL_GPUTransferBufferCreateInfo){.usage = SDL_GPU_TRANSFERBUFFERUSAGE_UPLOAD,
                                          .size = size},
        "texture-upload");
    if (!transfer)
    {
        SDL_LogError(SDL_LOG_CATEGORY_GPU, "Failed to create GPU transfer buffer: %s",
                     SDL_GetError());
        return false;
    }

    void* const mapped = SDL_MapGPUTransferBuffer(device, transfer, false);
    if (!mapped)
    {
        SDL_LogError(SDL_LOG_CATEGORY_GPU, "Failed to map GPU transfer buffer: %s", SDL_GetError());
        SDL_ReleaseGPUTransferBuffer(device, transfer);
        return false;
    }

    SDL_memcpy(mapped, data, size);
    SDL_UnmapGPUTransferBuffer(device, transfer);

    SDL_GPUCommandBuffer* const cmd = SDL_AcquireGPUCommandBuffer(device);
    if (!cmd)
    {
        SDL_LogError(SDL_LOG_CATEGORY_GPU, "Failed to acquire GPU command buffer: %s",
                     SDL_GetError());
        SDL_ReleaseGPUTransferBuffer(device, transfer);
        return false;
    }

    SDL_GPUCopyPass* const copy_pass = SDL_BeginGPUCopyPass(cmd);
    assert(copy_pass);
    SDL_UploadToGPUTexture(copy_pass,
                           &(SDL_GPUTextureTransferInfo){.transfer_buffer = transfer,
                                                         .offset = 0,
                                                         .pixels_per_row = grid_ext,
                                                         .rows_per_layer = grid_ext},
                           &(SDL_GPUTextureRegion){.texture = texture,
                                                   .mip_level = 0,
                                                   .layer = 0,
                                                   .x = 0,
                                                   .y = 0,
                                                   .z = 0,
                                                   .w = grid_ext,
                                                   .h = grid_ext,
                                                   .d = grid_ext},
                           false);
    SDL_EndGPUCopyPass(copy_pass);

    bool const submitted = SDL_SubmitGPUCommandBuffer(cmd);
    SDL_ReleaseGPUTransferBuffer(device, transfer);
    if (!submitted)
    {
        SDL_LogError(SDL_LOG_CATEGORY_GPU, "Failed to submit GPU upload command buffer: %s",
                     SDL_GetError());
        return false;
    }

    return true;
}

static SDL_GPUTexture* vx_create_occupancy_mask_texture(SDL_GPUDevice* const device,
                                                        vx_buffer(uint8_t) const masks,
                                                        int const         source_grid_ext,
                                                        char const* const name)
{
    int const mask_grid_ext = (source_grid_ext + VX_MASK_EXT - 1) / VX_MASK_EXT;
    assert(masks.ptr);
    assert(masks.count == mask_grid_ext * mask_grid_ext * mask_grid_ext);

    SDL_GPUTexture* const texture = vx_create_gpu_texture(
        device,
        (SDL_GPUTextureCreateInfo){.type = SDL_GPU_TEXTURETYPE_3D,
                                   .format = SDL_GPU_TEXTUREFORMAT_R8_UINT,
                                   .usage = SDL_GPU_TEXTUREUSAGE_GRAPHICS_STORAGE_READ |
                                            SDL_GPU_TEXTUREUSAGE_COMPUTE_STORAGE_READ,
                                   .width = (uint32_t)mask_grid_ext,
                                   .height = (uint32_t)mask_grid_ext,
                                   .layer_count_or_depth = (uint32_t)mask_grid_ext,
                                   .num_levels = 1,
                                   .sample_count = SDL_GPU_SAMPLECOUNT_1},
        name);
    if (!texture)
    {
        SDL_LogError(SDL_LOG_CATEGORY_GPU, "Failed to create %s texture: %s", name, SDL_GetError());
        return 0;
    }

    uint32_t const mask_buffer_size = (uint32_t)masks.count * sizeof(*masks.ptr);
    if (!vx_gpu_texture_upload(device, texture, masks.ptr, mask_buffer_size,
                               (uint32_t)mask_grid_ext))
    {
        SDL_ReleaseGPUTexture(device, texture);
        return 0;
    }

    return texture;
}

static SDL_GPUTexture* vx_create_aadf_texture(SDL_GPUDevice* const device, int const grid_ext,
                                              char const* const name)
{
    assert(grid_ext > 0);

    SDL_GPUTexture* const texture = vx_create_gpu_texture(
        device,
        (SDL_GPUTextureCreateInfo){.type = SDL_GPU_TEXTURETYPE_3D,
                                   .format = SDL_GPU_TEXTUREFORMAT_R32_UINT,
                                   .usage = SDL_GPU_TEXTUREUSAGE_GRAPHICS_STORAGE_READ |
                                            SDL_GPU_TEXTUREUSAGE_COMPUTE_STORAGE_READ |
                                            SDL_GPU_TEXTUREUSAGE_COMPUTE_STORAGE_WRITE,
                                   .width = (uint32_t)grid_ext,
                                   .height = (uint32_t)grid_ext,
                                   .layer_count_or_depth = (uint32_t)grid_ext,
                                   .num_levels = 1,
                                   .sample_count = SDL_GPU_SAMPLECOUNT_1},
        name);
    if (!texture)
    {
        SDL_LogError(SDL_LOG_CATEGORY_GPU, "Failed to create %s texture: %s", name, SDL_GetError());
        return 0;
    }

    return texture;
}

static void vx_dispatch_local_aadf(SDL_GPUCommandBuffer* const   cmd,
                                   SDL_GPUComputePipeline* const pipeline,
                                   SDL_GPUTexture* const         occupancy_masks,
                                   SDL_GPUTexture* const destination, uint32_t const grid_ext)
{
    SDL_GPUStorageTextureReadWriteBinding const destination_binding = {
        .texture = destination, .mip_level = 0, .layer = 0, .cycle = false};
    SDL_GPUComputePass* const pass = SDL_BeginGPUComputePass(cmd, &destination_binding, 1, 0, 0);
    assert(pass);
    SDL_BindGPUComputePipeline(pass, pipeline);
    SDL_BindGPUComputeStorageTextures(pass, 0, &occupancy_masks, 1);
    vx_aadf_uniforms const uniforms = {.grid_ext = grid_ext};
    SDL_PushGPUComputeUniformData(cmd, 0, &uniforms, sizeof(uniforms));
    uint32_t const group_count = (grid_ext + VX_BRICK_EXT - 1u) / VX_BRICK_EXT;
    SDL_DispatchGPUCompute(pass, group_count, group_count, group_count);
    SDL_EndGPUComputePass(pass);
}

static void vx_dispatch_global_aadf(SDL_GPUCommandBuffer* const   cmd,
                                    SDL_GPUComputePipeline* const pipeline,
                                    SDL_GPUTexture* const         occupancy_masks,
                                    SDL_GPUTexture* const source, SDL_GPUTexture* const destination,
                                    uint32_t const grid_ext, uint32_t const axis,
                                    bool const initialize)
{
    SDL_GPUStorageTextureReadWriteBinding const destination_binding = {
        .texture = destination, .mip_level = 0, .layer = 0, .cycle = false};
    SDL_GPUComputePass* const pass = SDL_BeginGPUComputePass(cmd, &destination_binding, 1, 0, 0);
    assert(pass);
    SDL_BindGPUComputePipeline(pass, pipeline);
    SDL_GPUTexture* const sources[] = {occupancy_masks, source};
    SDL_BindGPUComputeStorageTextures(pass, 0, sources, SDL_arraysize(sources));
    vx_aadf_uniforms const uniforms = {
        .grid_ext = grid_ext, .axis = axis, .initialize = initialize ? 1u : 0u};
    SDL_PushGPUComputeUniformData(cmd, 0, &uniforms, sizeof(uniforms));
    uint32_t const group_count = (grid_ext + VX_BRICK_EXT - 1u) / VX_BRICK_EXT;
    SDL_DispatchGPUCompute(pass, group_count, group_count, group_count);
    SDL_EndGPUComputePass(pass);
}

static bool
vx_create_gpu_aadfs(SDL_GPUDevice* const device, SDL_GPUTexture* const voxel_occupancy_masks,
                    SDL_GPUTexture* const brick_occupancy_masks,
                    SDL_GPUTexture* const chunk_occupancy_masks, uint32_t const grid_ext,
                    uint32_t const brick_grid_ext, uint32_t const chunk_grid_ext,
                    SDL_GPUTexture** const out_voxel_aadf, SDL_GPUTexture** const out_brick_aadf,
                    SDL_GPUTexture** const out_chunk_aadf)
{
    assert(device);
    assert(voxel_occupancy_masks);
    assert(brick_occupancy_masks);
    assert(chunk_occupancy_masks);
    assert(grid_ext > 0);
    assert(brick_grid_ext > 0);
    assert(chunk_grid_ext > 0);
    assert(out_voxel_aadf);
    assert(out_brick_aadf);
    assert(out_chunk_aadf);

    bool                  success = false;
    SDL_GPUTexture* const voxel_aadf = vx_create_aadf_texture(device, (int)grid_ext, "voxel-aadf");
    SDL_GPUTexture* const brick_aadf =
        vx_create_aadf_texture(device, (int)brick_grid_ext, "brick-aadf");
    SDL_GPUTexture* const chunk_aadf =
        vx_create_aadf_texture(device, (int)chunk_grid_ext, "chunk-aadf");
    SDL_GPUTexture* const chunk_aadf_scratch =
        vx_create_aadf_texture(device, (int)chunk_grid_ext, "chunk-aadf-scratch");
    if (!voxel_aadf || !brick_aadf || !chunk_aadf || !chunk_aadf_scratch)
    {
        SDL_LogError(SDL_LOG_CATEGORY_GPU, "Failed to create AADF textures: %s", SDL_GetError());
        goto cleanup_textures;
    }

    SDL_GPUComputePipeline* const local_pipeline = vx_create_gpu_compute_pipeline(
        device,
        (SDL_GPUComputePipelineCreateInfo){.code_size = AADF_LOCAL_CS_SIZE,
                                           .code = AADF_LOCAL_CS_BYTES,
                                           .entrypoint = GPU_SHADER_ENTRYPOINT,
                                           .format = GPU_SHADER_FORMAT,
                                           .num_readonly_storage_textures = 1,
                                           .num_readwrite_storage_textures = 1,
                                           .num_uniform_buffers = 1,
                                           .threadcount_x = VX_BRICK_EXT,
                                           .threadcount_y = VX_BRICK_EXT,
                                           .threadcount_z = VX_BRICK_EXT},
        "aadf-local");
    SDL_GPUComputePipeline* const global_pipeline = vx_create_gpu_compute_pipeline(
        device,
        (SDL_GPUComputePipelineCreateInfo){.code_size = AADF_GLOBAL_CS_SIZE,
                                           .code = AADF_GLOBAL_CS_BYTES,
                                           .entrypoint = GPU_SHADER_ENTRYPOINT,
                                           .format = GPU_SHADER_FORMAT,
                                           .num_readonly_storage_textures = 2,
                                           .num_readwrite_storage_textures = 1,
                                           .num_uniform_buffers = 1,
                                           .threadcount_x = VX_BRICK_EXT,
                                           .threadcount_y = VX_BRICK_EXT,
                                           .threadcount_z = VX_BRICK_EXT},
        "aadf-global");
    if (!local_pipeline || !global_pipeline)
    {
        SDL_LogError(SDL_LOG_CATEGORY_GPU, "Failed to create AADF pipelines: %s", SDL_GetError());
        goto cleanup_pipelines;
    }

    SDL_GPUCommandBuffer* const cmd = SDL_AcquireGPUCommandBuffer(device);
    if (!cmd)
    {
        SDL_LogError(SDL_LOG_CATEGORY_GPU, "Failed to acquire AADF command buffer: %s",
                     SDL_GetError());
        goto cleanup_pipelines;
    }

    vx_dispatch_local_aadf(cmd, local_pipeline, voxel_occupancy_masks, voxel_aadf, grid_ext);
    vx_dispatch_local_aadf(cmd, local_pipeline, brick_occupancy_masks, brick_aadf, brick_grid_ext);

    uint32_t const  propagation_rounds = chunk_grid_ext - 1u;
    SDL_GPUTexture* source = propagation_rounds % 2u != 0u ? chunk_aadf_scratch : chunk_aadf;
    SDL_GPUTexture* destination = source == chunk_aadf ? chunk_aadf_scratch : chunk_aadf;
    vx_dispatch_global_aadf(cmd, global_pipeline, chunk_occupancy_masks, destination, source,
                            chunk_grid_ext, 0, true);
    for (uint32_t round = 0; round < propagation_rounds; ++round)
    {
        for (uint32_t axis = 0; axis < 3u; ++axis)
        {
            vx_dispatch_global_aadf(cmd, global_pipeline, chunk_occupancy_masks, source,
                                    destination, chunk_grid_ext, axis, false);
            SDL_GPUTexture* const previous_source = source;
            source = destination;
            destination = previous_source;
        }
    }
    assert(source == chunk_aadf);

    if (!SDL_SubmitGPUCommandBuffer(cmd))
    {
        SDL_LogError(SDL_LOG_CATEGORY_GPU, "Failed to submit AADF command buffer: %s",
                     SDL_GetError());
        goto cleanup_pipelines;
    }

    *out_voxel_aadf = voxel_aadf;
    *out_brick_aadf = brick_aadf;
    *out_chunk_aadf = chunk_aadf;
    success = true;

cleanup_pipelines:
    if (global_pipeline)
    {
        SDL_ReleaseGPUComputePipeline(device, global_pipeline);
    }
    if (local_pipeline)
    {
        SDL_ReleaseGPUComputePipeline(device, local_pipeline);
    }
cleanup_textures:
    if (chunk_aadf_scratch)
    {
        SDL_ReleaseGPUTexture(device, chunk_aadf_scratch);
    }
    if (!success)
    {
        if (chunk_aadf)
        {
            SDL_ReleaseGPUTexture(device, chunk_aadf);
        }
        if (brick_aadf)
        {
            SDL_ReleaseGPUTexture(device, brick_aadf);
        }
        if (voxel_aadf)
        {
            SDL_ReleaseGPUTexture(device, voxel_aadf);
        }
    }
    return success;
}

typedef struct vx_camera
{
    vec3s position;
    float yaw;
    float pitch;
} vx_camera;

static void vx_camera_print_code(vx_camera const* const camera)
{
    printf(
        "position = %.9g %.9g %.9g\n"
        "yaw = %.9g\n"
        "pitch = %.9g\n",
        camera->position.x, camera->position.y, camera->position.z, camera->yaw, camera->pitch);
}

static bool vx_camera_load(vx_camera* const camera, char const* const path)
{
    FILE* const file = fopen(path, "r");
    if (!file)
    {
        SDL_LogError(SDL_LOG_CATEGORY_APPLICATION, "Couldn't open camera preset %s\n", path);
        return false;
    }

    vx_camera  loaded_camera = {0};
    int const  parsed = fscanf(file, "position = %f %f %f\nyaw = %f\npitch = %f",
                               &loaded_camera.position.x, &loaded_camera.position.y,
                               &loaded_camera.position.z, &loaded_camera.yaw, &loaded_camera.pitch);
    char       extra = 0;
    bool const has_extra_data = fscanf(file, " %c", &extra) != EOF;
    bool const closed = fclose(file) == 0;
    if (parsed != 5 || has_extra_data || !closed || !isfinite(loaded_camera.position.x) ||
        !isfinite(loaded_camera.position.y) || !isfinite(loaded_camera.position.z) ||
        !isfinite(loaded_camera.yaw) || !isfinite(loaded_camera.pitch))
    {
        SDL_LogError(SDL_LOG_CATEGORY_APPLICATION, "Invalid camera preset %s\n", path);
        return false;
    }

    *camera = loaded_camera;
    return true;
}

static vec3s vx_camera_forward(vx_camera const* const camera)
{
    float const cos_pitch = cosf(camera->pitch);
    return glms_vec3_normalize(
        (vec3s){sinf(camera->yaw) * cos_pitch, sinf(camera->pitch), cosf(camera->yaw) * cos_pitch});
}

static void vx_camera_basis(vx_camera const* const camera, vec3s* const right, vec3s* const up,
                            vec3s* const forward)
{
    *forward = vx_camera_forward(camera);

    vec3s const world_up = {0.f, 1.f, 0.f};
    *right = glms_vec3_crossn(world_up, *forward);
    *up = glms_vec3_cross(*forward, *right);
}

static void vx_camera_look_at(vx_camera* const camera, vec3s const target)
{
    vec3s const forward = glms_vec3_normalize(glms_vec3_sub(target, camera->position));
    camera->pitch = asinf(SDL_clamp(forward.y, -1.f, 1.f));
    camera->yaw = atan2f(forward.x, forward.z);
}

static void vx_camera_update_movement(vx_camera* const camera)
{
    vec3s right = {0};
    vec3s up = {0};
    vec3s forward = {0};
    vx_camera_basis(camera, &right, &up, &forward);

    bool const* const keys = SDL_GetKeyboardState(0);
    float const       move_speed = 1.0f;
    vec3s             move = {0};
    if (keys[SDL_SCANCODE_W])
    {
        move = glms_vec3_add(move, forward);
    }
    if (keys[SDL_SCANCODE_S])
    {
        move = glms_vec3_sub(move, forward);
    }
    if (keys[SDL_SCANCODE_D])
    {
        move = glms_vec3_add(move, right);
    }
    if (keys[SDL_SCANCODE_A])
    {
        move = glms_vec3_sub(move, right);
    }
    if (keys[SDL_SCANCODE_E])
    {
        move = glms_vec3_add(move, up);
    }
    if (keys[SDL_SCANCODE_Q])
    {
        move = glms_vec3_sub(move, up);
    }
    if (glms_vec3_norm2(move) > 0.f)
    {
        camera->position =
            glms_vec3_add(camera->position, glms_vec3_scale(glms_vec3_normalize(move), move_speed));
    }
}

enum
{
    VX_INPUT_POINTER_DOWN = 1u << 0,
    VX_INPUT_POINTER_PRESSED = 1u << 1,
    VX_INPUT_POINTER_UP = 1u << 2,
};

enum
{
    VX_RTAO_MAX_SAMPLES_PER_FRAME = 32,
};

typedef struct vx_input
{
    unsigned int pointer_events;
    float        pointer_delta[2];
} vx_input;

typedef struct vxray
{
    // Platform
    SDL_GPUDevice* gpu_device;
    SDL_Window*    window;
    bool           window_claimed;

    // Camera
    vx_camera camera;
    vx_input  input;
    int       display_texture;
    float     rtao_radius;
    int       rtao_samples_per_frame;
    float     ao_sp;
    float     ao_smin;
    float     sun_elevation_degrees;
    float     sun_azimuth_degrees;
    float     view_altitude_km;
    bool      sky_view_dirty;
    uint32_t  frame_index;
    float4x4  previous_view_projection;
    bool      rtao_history_valid;

    // Voxel grid
    int grid_ext;

    // GPU
    SDL_GPUGraphicsPipeline* gbuffer_pipeline;
    SDL_GPUGraphicsPipeline* rtao_index_pipeline;
    SDL_GPUGraphicsPipeline* rtao_pipeline;
    SDL_GPUGraphicsPipeline* sky_view_pipeline;
    SDL_GPUGraphicsPipeline* display_pipeline;
    SDL_GPUTexture*          voxel_texture;
    SDL_GPUTexture*          voxel_mask_texture;
    SDL_GPUTexture*          brick_mask_texture;
    SDL_GPUTexture*          chunk_mask_texture;
    SDL_GPUTexture*          voxel_aadf_texture;
    SDL_GPUTexture*          brick_aadf_texture;
    SDL_GPUTexture*          chunk_aadf_texture;
    SDL_GPUTexture*          gbuffer_albedo_texture;
    SDL_GPUTexture*          gbuffer_normal_texture;
    SDL_GPUTexture*          gbuffer_depth_texture;
    SDL_GPUTexture*          rtao_index_textures[2];
    SDL_GPUTexture*          rtao_checksum_textures[2];
    SDL_GPUTexture*          rtao_visibility_texture;
    SDL_GPUTexture*          sky_view_texture;
    uint32_t                 render_width;
    uint32_t                 render_height;
    SDL_GPUSampler*          display_sampler;
    SDL_GPUSampler*          sky_view_sampler;
    SDL_GPUBuffer*           palette_buffer;
    SDL_GPUBuffer*           ao_checksum_buffer;
    SDL_GPUBuffer*           ao_payload_buffer;
    SDL_GPUBuffer*           ao_last_touched_frame_buffer;
    SDL_GPUTransferBuffer*   ao_reset_transfer_buffer;
} vxray;

static vxray vxray_instance = {0};

static bool vx_ensure_render_textures(uint32_t const width, uint32_t const height)
{
    assert(width > 0);
    assert(height > 0);

    if (vxray_instance.gbuffer_albedo_texture && vxray_instance.gbuffer_normal_texture &&
        vxray_instance.gbuffer_depth_texture && vxray_instance.rtao_index_textures[0] &&
        vxray_instance.rtao_index_textures[1] && vxray_instance.rtao_checksum_textures[0] &&
        vxray_instance.rtao_checksum_textures[1] && vxray_instance.rtao_visibility_texture &&
        vxray_instance.render_width == width && vxray_instance.render_height == height)
    {
        return true;
    }

    SDL_GPUTexture* const gbuffer_albedo_texture = vx_create_gpu_texture(
        vxray_instance.gpu_device,
        (SDL_GPUTextureCreateInfo){.type = SDL_GPU_TEXTURETYPE_2D,
                                   .format = SDL_GPU_TEXTUREFORMAT_R32_UINT,
                                   .usage = SDL_GPU_TEXTUREUSAGE_COLOR_TARGET |
                                            SDL_GPU_TEXTUREUSAGE_GRAPHICS_STORAGE_READ,
                                   .width = width,
                                   .height = height,
                                   .layer_count_or_depth = 1,
                                   .num_levels = 1,
                                   .sample_count = SDL_GPU_SAMPLECOUNT_1},
        "gbuffer-albedo");
    if (!gbuffer_albedo_texture)
    {
        SDL_LogError(SDL_LOG_CATEGORY_GPU, "Failed to create G-buffer albedo texture: %s",
                     SDL_GetError());
        return false;
    }

    SDL_GPUTexture* const gbuffer_normal_texture = vx_create_gpu_texture(
        vxray_instance.gpu_device,
        (SDL_GPUTextureCreateInfo){.type = SDL_GPU_TEXTURETYPE_2D,
                                   .format = SDL_GPU_TEXTUREFORMAT_R8_UINT,
                                   .usage = SDL_GPU_TEXTUREUSAGE_COLOR_TARGET |
                                            SDL_GPU_TEXTUREUSAGE_GRAPHICS_STORAGE_READ,
                                   .width = width,
                                   .height = height,
                                   .layer_count_or_depth = 1,
                                   .num_levels = 1,
                                   .sample_count = SDL_GPU_SAMPLECOUNT_1},
        "gbuffer-normal");
    if (!gbuffer_normal_texture)
    {
        SDL_LogError(SDL_LOG_CATEGORY_GPU, "Failed to create G-buffer normal texture: %s",
                     SDL_GetError());
        SDL_ReleaseGPUTexture(vxray_instance.gpu_device, gbuffer_albedo_texture);
        return false;
    }

    SDL_GPUTexture* const gbuffer_depth_texture = vx_create_gpu_texture(
        vxray_instance.gpu_device,
        (SDL_GPUTextureCreateInfo){.type = SDL_GPU_TEXTURETYPE_2D,
                                   .format = SDL_GPU_TEXTUREFORMAT_D32_FLOAT,
                                   .usage = SDL_GPU_TEXTUREUSAGE_DEPTH_STENCIL_TARGET |
                                            SDL_GPU_TEXTUREUSAGE_SAMPLER,
                                   .width = width,
                                   .height = height,
                                   .layer_count_or_depth = 1,
                                   .num_levels = 1,
                                   .sample_count = SDL_GPU_SAMPLECOUNT_1},
        "gbuffer-depth");
    if (!gbuffer_depth_texture)
    {
        SDL_LogError(SDL_LOG_CATEGORY_GPU, "Failed to create G-buffer depth texture: %s",
                     SDL_GetError());
        SDL_ReleaseGPUTexture(vxray_instance.gpu_device, gbuffer_normal_texture);
        SDL_ReleaseGPUTexture(vxray_instance.gpu_device, gbuffer_albedo_texture);
        return false;
    }

    SDL_GPUTextureCreateInfo const rtao_history_texture_info = {
        .type = SDL_GPU_TEXTURETYPE_2D,
        .format = SDL_GPU_TEXTUREFORMAT_R32_UINT,
        .usage = SDL_GPU_TEXTUREUSAGE_COLOR_TARGET | SDL_GPU_TEXTUREUSAGE_GRAPHICS_STORAGE_READ,
        .width = width,
        .height = height,
        .layer_count_or_depth = 1,
        .num_levels = 1,
        .sample_count = SDL_GPU_SAMPLECOUNT_1};
    char const* const rtao_index_texture_names[] = {"rtao-index-0", "rtao-index-1"};
    char const* const rtao_checksum_texture_names[] = {"rtao-checksum-0", "rtao-checksum-1"};
    SDL_GPUTexture*   rtao_index_textures[2] = {0};
    SDL_GPUTexture*   rtao_checksum_textures[2] = {0};
    for (uint32_t i = 0u; i < 2u; ++i)
    {
        rtao_index_textures[i] = vx_create_gpu_texture(
            vxray_instance.gpu_device, rtao_history_texture_info, rtao_index_texture_names[i]);
        rtao_checksum_textures[i] = vx_create_gpu_texture(
            vxray_instance.gpu_device, rtao_history_texture_info, rtao_checksum_texture_names[i]);
        if (!rtao_index_textures[i] || !rtao_checksum_textures[i])
        {
            SDL_LogError(SDL_LOG_CATEGORY_GPU, "Failed to create RTAO history textures: %s",
                         SDL_GetError());
            for (uint32_t j = 0u; j < 2u; ++j)
            {
                if (rtao_checksum_textures[j])
                {
                    SDL_ReleaseGPUTexture(vxray_instance.gpu_device, rtao_checksum_textures[j]);
                }
                if (rtao_index_textures[j])
                {
                    SDL_ReleaseGPUTexture(vxray_instance.gpu_device, rtao_index_textures[j]);
                }
            }
            SDL_ReleaseGPUTexture(vxray_instance.gpu_device, gbuffer_depth_texture);
            SDL_ReleaseGPUTexture(vxray_instance.gpu_device, gbuffer_normal_texture);
            SDL_ReleaseGPUTexture(vxray_instance.gpu_device, gbuffer_albedo_texture);
            return false;
        }
    }

    SDL_GPUTexture* const rtao_visibility_texture = vx_create_gpu_texture(
        vxray_instance.gpu_device,
        (SDL_GPUTextureCreateInfo){.type = SDL_GPU_TEXTURETYPE_2D,
                                   .format = SDL_GPU_TEXTUREFORMAT_R16_FLOAT,
                                   .usage = SDL_GPU_TEXTUREUSAGE_COLOR_TARGET |
                                            SDL_GPU_TEXTUREUSAGE_SAMPLER,
                                   .width = width,
                                   .height = height,
                                   .layer_count_or_depth = 1,
                                   .num_levels = 1,
                                   .sample_count = SDL_GPU_SAMPLECOUNT_1},
        "rtao-visibility");
    if (!rtao_visibility_texture)
    {
        SDL_LogError(SDL_LOG_CATEGORY_GPU, "Failed to create RTAO visibility texture: %s",
                     SDL_GetError());
        for (uint32_t i = 0u; i < 2u; ++i)
        {
            SDL_ReleaseGPUTexture(vxray_instance.gpu_device, rtao_checksum_textures[i]);
            SDL_ReleaseGPUTexture(vxray_instance.gpu_device, rtao_index_textures[i]);
        }
        SDL_ReleaseGPUTexture(vxray_instance.gpu_device, gbuffer_depth_texture);
        SDL_ReleaseGPUTexture(vxray_instance.gpu_device, gbuffer_normal_texture);
        SDL_ReleaseGPUTexture(vxray_instance.gpu_device, gbuffer_albedo_texture);
        return false;
    }

    if (vxray_instance.gbuffer_albedo_texture)
    {
        SDL_ReleaseGPUTexture(vxray_instance.gpu_device, vxray_instance.gbuffer_albedo_texture);
    }
    if (vxray_instance.gbuffer_normal_texture)
    {
        SDL_ReleaseGPUTexture(vxray_instance.gpu_device, vxray_instance.gbuffer_normal_texture);
    }
    if (vxray_instance.gbuffer_depth_texture)
    {
        SDL_ReleaseGPUTexture(vxray_instance.gpu_device, vxray_instance.gbuffer_depth_texture);
    }
    for (uint32_t i = 0u; i < 2u; ++i)
    {
        if (vxray_instance.rtao_checksum_textures[i])
        {
            SDL_ReleaseGPUTexture(vxray_instance.gpu_device,
                                  vxray_instance.rtao_checksum_textures[i]);
        }
        if (vxray_instance.rtao_index_textures[i])
        {
            SDL_ReleaseGPUTexture(vxray_instance.gpu_device, vxray_instance.rtao_index_textures[i]);
        }
    }
    if (vxray_instance.rtao_visibility_texture)
    {
        SDL_ReleaseGPUTexture(vxray_instance.gpu_device, vxray_instance.rtao_visibility_texture);
    }
    vxray_instance.gbuffer_albedo_texture = gbuffer_albedo_texture;
    vxray_instance.gbuffer_normal_texture = gbuffer_normal_texture;
    vxray_instance.gbuffer_depth_texture = gbuffer_depth_texture;
    for (uint32_t i = 0u; i < 2u; ++i)
    {
        vxray_instance.rtao_index_textures[i] = rtao_index_textures[i];
        vxray_instance.rtao_checksum_textures[i] = rtao_checksum_textures[i];
    }
    vxray_instance.rtao_visibility_texture = rtao_visibility_texture;
    vxray_instance.render_width = width;
    vxray_instance.render_height = height;
    vxray_instance.rtao_history_valid = false;
    return true;
}

SDL_AppResult SDL_AppInit(void** const appstate, int const argc, char* argv[])
{
    (void)appstate;

    vxray_instance.rtao_radius = 8.f;
    vxray_instance.rtao_samples_per_frame = 1;
    vxray_instance.ao_sp = 10.f;
    vxray_instance.ao_smin = 0.07f;
    vxray_instance.sun_elevation_degrees = 10.f;
    vxray_instance.sun_azimuth_degrees = 180.f;
    vxray_instance.view_altitude_km = 1.f;
    vxray_instance.sky_view_dirty = true;

    if (argc < 2 || argc > 3)
    {
        fprintf(stderr, "Usage: %s <file.vox> [camera.vx]\n", argv[0]);
        return SDL_APP_FAILURE;
    }

    // Init

    if (!SDL_Init(SDL_INIT_VIDEO))
    {
        SDL_LogError(SDL_LOG_CATEGORY_VIDEO, "Failed to initialize video: %s", SDL_GetError());
        return SDL_APP_FAILURE;
    }

    // Window

    {
        SDL_WindowFlags const flags = SDL_WINDOW_RESIZABLE | SDL_WINDOW_HIGH_PIXEL_DENSITY;
        SDL_Window* const     window = SDL_CreateWindow("vxray", 640, 480, flags);
        if (!window)
        {
            SDL_LogError(SDL_LOG_CATEGORY_VIDEO, "Failed to create window: %s", SDL_GetError());
            return SDL_APP_FAILURE;
        }

        vxray_instance.window = window;
    }

    // GPU device

    {
        SDL_GPUDevice* const gpu_device =
            SDL_CreateGPUDevice(GPU_SHADER_FORMAT, GPU_DEVICE_DEBUG_MODE, GPU_DRIVER_NAME);

        if (!gpu_device)
        {
            SDL_LogError(SDL_LOG_CATEGORY_GPU, "Failed to create GPU device: %s", SDL_GetError());
            return SDL_APP_FAILURE;
        }

        vxray_instance.gpu_device = gpu_device;
    }

    if (!SDL_ClaimWindowForGPUDevice(vxray_instance.gpu_device, vxray_instance.window))
    {
        SDL_LogError(SDL_LOG_CATEGORY_GPU, "Couldn't claim window for GPU device: %s",
                     SDL_GetError());
        return SDL_APP_FAILURE;
    }
    vxray_instance.window_claimed = true;
    if (!SDL_SetGPUSwapchainParameters(vxray_instance.gpu_device, vxray_instance.window,
                                       SDL_GPU_SWAPCHAINCOMPOSITION_SDR_LINEAR,
                                       SDL_GPU_PRESENTMODE_VSYNC))
    {
        SDL_LogError(SDL_LOG_CATEGORY_GPU, "Couldn't set GPU swapchain parameters: %s",
                     SDL_GetError());
        return SDL_APP_FAILURE;
    }

    if (!imgui_sdl3_init(vxray_instance.gpu_device, vxray_instance.window))
    {
        SDL_LogError(SDL_LOG_CATEGORY_GPU, "Couldn't initialize ImGui: %s", SDL_GetError());
        return SDL_APP_FAILURE;
    }

    if (!SDL_GPUTextureSupportsFormat(
            vxray_instance.gpu_device, SDL_GPU_TEXTUREFORMAT_D32_FLOAT, SDL_GPU_TEXTURETYPE_2D,
            SDL_GPU_TEXTUREUSAGE_DEPTH_STENCIL_TARGET | SDL_GPU_TEXTUREUSAGE_SAMPLER))
    {
        SDL_LogError(SDL_LOG_CATEGORY_GPU, "TEXTUREFORMAT_D32_FLOAT not supported on this device");
        return SDL_APP_FAILURE;
    }
    if (!SDL_GPUTextureSupportsFormat(
            vxray_instance.gpu_device, SDL_GPU_TEXTUREFORMAT_R32_UINT, SDL_GPU_TEXTURETYPE_2D,
            SDL_GPU_TEXTUREUSAGE_COLOR_TARGET | SDL_GPU_TEXTUREUSAGE_GRAPHICS_STORAGE_READ))
    {
        SDL_LogError(SDL_LOG_CATEGORY_GPU, "TEXTUREFORMAT_R32_UINT not supported on this device");
        return SDL_APP_FAILURE;
    }
    if (!SDL_GPUTextureSupportsFormat(
            vxray_instance.gpu_device, SDL_GPU_TEXTUREFORMAT_R8_UINT, SDL_GPU_TEXTURETYPE_2D,
            SDL_GPU_TEXTUREUSAGE_COLOR_TARGET | SDL_GPU_TEXTUREUSAGE_GRAPHICS_STORAGE_READ))
    {
        SDL_LogError(SDL_LOG_CATEGORY_GPU, "TEXTUREFORMAT_R8_UINT not supported on this device");
        return SDL_APP_FAILURE;
    }
    if (!SDL_GPUTextureSupportsFormat(
            vxray_instance.gpu_device, SDL_GPU_TEXTUREFORMAT_R16_FLOAT, SDL_GPU_TEXTURETYPE_2D,
            SDL_GPU_TEXTUREUSAGE_COLOR_TARGET | SDL_GPU_TEXTUREUSAGE_SAMPLER))
    {
        SDL_LogError(SDL_LOG_CATEGORY_GPU, "TEXTUREFORMAT_R16_FLOAT not supported on this device");
        return SDL_APP_FAILURE;
    }
    if (!SDL_GPUTextureSupportsFormat(
            vxray_instance.gpu_device, SDL_GPU_TEXTUREFORMAT_R16G16B16A16_FLOAT,
            SDL_GPU_TEXTURETYPE_2D,
            SDL_GPU_TEXTUREUSAGE_COLOR_TARGET | SDL_GPU_TEXTUREUSAGE_SAMPLER))
    {
        SDL_LogError(SDL_LOG_CATEGORY_GPU,
                     "TEXTUREFORMAT_R16G16B16A16_FLOAT not supported on this device");
        return SDL_APP_FAILURE;
    }

    // G-buffer fullscreen pipeline

    {
        SDL_GPUShaderCreateInfo const vs_info = {.code_size = FULLSCREEN_VS_SIZE,
                                                 .code = FULLSCREEN_VS_BYTES,
                                                 .entrypoint = GPU_SHADER_ENTRYPOINT,
                                                 .format = GPU_SHADER_FORMAT,
                                                 .stage = SDL_GPU_SHADERSTAGE_VERTEX};
        SDL_GPUShaderCreateInfo const ps_info = {.code_size = GBUFFER_PS_SIZE,
                                                 .code = GBUFFER_PS_BYTES,
                                                 .entrypoint = GPU_SHADER_ENTRYPOINT,
                                                 .format = GPU_SHADER_FORMAT,
                                                 .stage = SDL_GPU_SHADERSTAGE_FRAGMENT,
                                                 .num_storage_textures = 7,
                                                 .num_storage_buffers = 1,
                                                 .num_uniform_buffers = 1};
        SDL_GPUShader* const          vertex_shader =
            SDL_CreateGPUShader(vxray_instance.gpu_device, &vs_info);
        SDL_GPUShader* const fragment_shader =
            SDL_CreateGPUShader(vxray_instance.gpu_device, &ps_info);
        if (!vertex_shader || !fragment_shader)
        {
            SDL_LogError(SDL_LOG_CATEGORY_GPU, "Couldn't create G-buffer shaders: %s",
                         SDL_GetError());
            if (fragment_shader)
            {
                SDL_ReleaseGPUShader(vxray_instance.gpu_device, fragment_shader);
            }
            if (vertex_shader)
            {
                SDL_ReleaseGPUShader(vxray_instance.gpu_device, vertex_shader);
            }
            return SDL_APP_FAILURE;
        }

        vxray_instance.gbuffer_pipeline = vx_create_gpu_graphics_pipeline(
            vxray_instance.gpu_device,
            (SDL_GPUGraphicsPipelineCreateInfo){
                .vertex_shader = vertex_shader,
                .fragment_shader = fragment_shader,
                .primitive_type = SDL_GPU_PRIMITIVETYPE_TRIANGLELIST,
                .rasterizer_state =
                    (SDL_GPURasterizerState){.fill_mode = SDL_GPU_FILLMODE_FILL,
                                             .cull_mode = SDL_GPU_CULLMODE_BACK,
                                             .front_face = SDL_GPU_FRONTFACE_CLOCKWISE},
                .depth_stencil_state =
                    (SDL_GPUDepthStencilState){.compare_op = SDL_GPU_COMPAREOP_ALWAYS,
                                               .enable_depth_test = true,
                                               .enable_depth_write = true},
                .target_info =
                    (SDL_GPUGraphicsPipelineTargetInfo){
                        .num_color_targets = 2,
                        .color_target_descriptions =
                            (SDL_GPUColorTargetDescription[]){
                                {.format = SDL_GPU_TEXTUREFORMAT_R32_UINT},
                                {.format = SDL_GPU_TEXTUREFORMAT_R8_UINT}},
                        .depth_stencil_format = SDL_GPU_TEXTUREFORMAT_D32_FLOAT,
                        .has_depth_stencil_target = true}},
            "gbuffer-raster");
        SDL_ReleaseGPUShader(vxray_instance.gpu_device, fragment_shader);
        SDL_ReleaseGPUShader(vxray_instance.gpu_device, vertex_shader);
        if (!vxray_instance.gbuffer_pipeline)
        {
            SDL_LogError(SDL_LOG_CATEGORY_GPU, "Couldn't create G-buffer pipeline: %s",
                         SDL_GetError());
            return SDL_APP_FAILURE;
        }
    }

    // RTAO spatial-cache index pipeline

    {
        SDL_GPUShaderCreateInfo const vs_info = {.code_size = FULLSCREEN_VS_SIZE,
                                                 .code = FULLSCREEN_VS_BYTES,
                                                 .entrypoint = GPU_SHADER_ENTRYPOINT,
                                                 .format = GPU_SHADER_FORMAT,
                                                 .stage = SDL_GPU_SHADERSTAGE_VERTEX};
        SDL_GPUShaderCreateInfo const ps_info = {.code_size = RTAO_INDEX_PS_SIZE,
                                                 .code = RTAO_INDEX_PS_BYTES,
                                                 .entrypoint = GPU_SHADER_ENTRYPOINT,
                                                 .format = GPU_SHADER_FORMAT,
                                                 .stage = SDL_GPU_SHADERSTAGE_FRAGMENT,
                                                 .num_samplers = 1,
                                                 .num_storage_textures = 3,
                                                 .num_storage_buffers = 3,
                                                 .num_uniform_buffers = 1};
        SDL_GPUShader* const          vertex_shader =
            SDL_CreateGPUShader(vxray_instance.gpu_device, &vs_info);
        SDL_GPUShader* const fragment_shader =
            SDL_CreateGPUShader(vxray_instance.gpu_device, &ps_info);
        if (!vertex_shader || !fragment_shader)
        {
            SDL_LogError(SDL_LOG_CATEGORY_GPU, "Couldn't create RTAO index shaders: %s",
                         SDL_GetError());
            if (fragment_shader)
            {
                SDL_ReleaseGPUShader(vxray_instance.gpu_device, fragment_shader);
            }
            if (vertex_shader)
            {
                SDL_ReleaseGPUShader(vxray_instance.gpu_device, vertex_shader);
            }
            return SDL_APP_FAILURE;
        }

        vxray_instance.rtao_index_pipeline = vx_create_gpu_graphics_pipeline(
            vxray_instance.gpu_device,
            (SDL_GPUGraphicsPipelineCreateInfo){
                .vertex_shader = vertex_shader,
                .fragment_shader = fragment_shader,
                .primitive_type = SDL_GPU_PRIMITIVETYPE_TRIANGLELIST,
                .rasterizer_state =
                    (SDL_GPURasterizerState){.fill_mode = SDL_GPU_FILLMODE_FILL,
                                             .cull_mode = SDL_GPU_CULLMODE_BACK,
                                             .front_face = SDL_GPU_FRONTFACE_CLOCKWISE},
                .target_info =
                    (SDL_GPUGraphicsPipelineTargetInfo){
                        .num_color_targets = 2,
                        .color_target_descriptions =
                            (SDL_GPUColorTargetDescription[]){
                                {.format = SDL_GPU_TEXTUREFORMAT_R32_UINT},
                                {.format = SDL_GPU_TEXTUREFORMAT_R32_UINT}}}},
            "rtao-index-raster");
        SDL_ReleaseGPUShader(vxray_instance.gpu_device, fragment_shader);
        SDL_ReleaseGPUShader(vxray_instance.gpu_device, vertex_shader);
        if (!vxray_instance.rtao_index_pipeline)
        {
            SDL_LogError(SDL_LOG_CATEGORY_GPU, "Couldn't create RTAO index pipeline: %s",
                         SDL_GetError());
            return SDL_APP_FAILURE;
        }
    }

    // RTAO sample pipeline

    {
        SDL_GPUShaderCreateInfo const vs_info = {.code_size = FULLSCREEN_VS_SIZE,
                                                 .code = FULLSCREEN_VS_BYTES,
                                                 .entrypoint = GPU_SHADER_ENTRYPOINT,
                                                 .format = GPU_SHADER_FORMAT,
                                                 .stage = SDL_GPU_SHADERSTAGE_VERTEX};
        SDL_GPUShaderCreateInfo const ps_info = {.code_size = RTAO_PS_SIZE,
                                                 .code = RTAO_PS_BYTES,
                                                 .entrypoint = GPU_SHADER_ENTRYPOINT,
                                                 .format = GPU_SHADER_FORMAT,
                                                 .stage = SDL_GPU_SHADERSTAGE_FRAGMENT,
                                                 .num_samplers = 1,
                                                 .num_storage_textures = 4,
                                                 .num_storage_buffers = 1,
                                                 .num_uniform_buffers = 1};
        SDL_GPUShader* const          vertex_shader =
            SDL_CreateGPUShader(vxray_instance.gpu_device, &vs_info);
        SDL_GPUShader* const fragment_shader =
            SDL_CreateGPUShader(vxray_instance.gpu_device, &ps_info);
        if (!vertex_shader || !fragment_shader)
        {
            SDL_LogError(SDL_LOG_CATEGORY_GPU, "Couldn't create RTAO visibility shaders: %s",
                         SDL_GetError());
            if (fragment_shader)
            {
                SDL_ReleaseGPUShader(vxray_instance.gpu_device, fragment_shader);
            }
            if (vertex_shader)
            {
                SDL_ReleaseGPUShader(vxray_instance.gpu_device, vertex_shader);
            }
            return SDL_APP_FAILURE;
        }

        vxray_instance.rtao_pipeline = vx_create_gpu_graphics_pipeline(
            vxray_instance.gpu_device,
            (SDL_GPUGraphicsPipelineCreateInfo){
                .vertex_shader = vertex_shader,
                .fragment_shader = fragment_shader,
                .primitive_type = SDL_GPU_PRIMITIVETYPE_TRIANGLELIST,
                .rasterizer_state =
                    (SDL_GPURasterizerState){.fill_mode = SDL_GPU_FILLMODE_FILL,
                                             .cull_mode = SDL_GPU_CULLMODE_BACK,
                                             .front_face = SDL_GPU_FRONTFACE_CLOCKWISE},
                .target_info =
                    (SDL_GPUGraphicsPipelineTargetInfo){
                        .num_color_targets = 1,
                        .color_target_descriptions =
                            (SDL_GPUColorTargetDescription[]){
                                {.format = SDL_GPU_TEXTUREFORMAT_R16_FLOAT}}}},
            "rtao-raster");
        SDL_ReleaseGPUShader(vxray_instance.gpu_device, fragment_shader);
        SDL_ReleaseGPUShader(vxray_instance.gpu_device, vertex_shader);
        if (!vxray_instance.rtao_pipeline)
        {
            SDL_LogError(SDL_LOG_CATEGORY_GPU, "Couldn't create RTAO pipeline: %s", SDL_GetError());
            return SDL_APP_FAILURE;
        }
    }

    // Sky-view LUT pipeline

    {
        SDL_GPUShaderCreateInfo const vs_info = {.code_size = FULLSCREEN_VS_SIZE,
                                                 .code = FULLSCREEN_VS_BYTES,
                                                 .entrypoint = GPU_SHADER_ENTRYPOINT,
                                                 .format = GPU_SHADER_FORMAT,
                                                 .stage = SDL_GPU_SHADERSTAGE_VERTEX};
        SDL_GPUShaderCreateInfo const ps_info = {.code_size = SKY_VIEW_PS_SIZE,
                                                 .code = SKY_VIEW_PS_BYTES,
                                                 .entrypoint = GPU_SHADER_ENTRYPOINT,
                                                 .format = GPU_SHADER_FORMAT,
                                                 .stage = SDL_GPU_SHADERSTAGE_FRAGMENT,
                                                 .num_uniform_buffers = 1};
        SDL_GPUShader* const          vertex_shader =
            SDL_CreateGPUShader(vxray_instance.gpu_device, &vs_info);
        SDL_GPUShader* const fragment_shader =
            SDL_CreateGPUShader(vxray_instance.gpu_device, &ps_info);
        if (!vertex_shader || !fragment_shader)
        {
            SDL_LogError(SDL_LOG_CATEGORY_GPU, "Couldn't create sky-view shaders: %s",
                         SDL_GetError());
            if (fragment_shader)
            {
                SDL_ReleaseGPUShader(vxray_instance.gpu_device, fragment_shader);
            }
            if (vertex_shader)
            {
                SDL_ReleaseGPUShader(vxray_instance.gpu_device, vertex_shader);
            }
            return SDL_APP_FAILURE;
        }

        vxray_instance.sky_view_pipeline = vx_create_gpu_graphics_pipeline(
            vxray_instance.gpu_device,
            (SDL_GPUGraphicsPipelineCreateInfo){
                .vertex_shader = vertex_shader,
                .fragment_shader = fragment_shader,
                .primitive_type = SDL_GPU_PRIMITIVETYPE_TRIANGLELIST,
                .rasterizer_state =
                    (SDL_GPURasterizerState){.fill_mode = SDL_GPU_FILLMODE_FILL,
                                             .cull_mode = SDL_GPU_CULLMODE_BACK,
                                             .front_face = SDL_GPU_FRONTFACE_CLOCKWISE},
                .target_info =
                    (SDL_GPUGraphicsPipelineTargetInfo){
                        .num_color_targets = 1,
                        .color_target_descriptions =
                            (SDL_GPUColorTargetDescription[]){
                                {.format = SDL_GPU_TEXTUREFORMAT_R16G16B16A16_FLOAT}}}},
            "sky-view-raster");
        SDL_ReleaseGPUShader(vxray_instance.gpu_device, fragment_shader);
        SDL_ReleaseGPUShader(vxray_instance.gpu_device, vertex_shader);
        if (!vxray_instance.sky_view_pipeline)
        {
            SDL_LogError(SDL_LOG_CATEGORY_GPU, "Couldn't create sky-view pipeline: %s",
                         SDL_GetError());
            return SDL_APP_FAILURE;
        }
    }

    vxray_instance.sky_view_texture = vx_create_gpu_texture(
        vxray_instance.gpu_device,
        (SDL_GPUTextureCreateInfo){.type = SDL_GPU_TEXTURETYPE_2D,
                                   .format = SDL_GPU_TEXTUREFORMAT_R16G16B16A16_FLOAT,
                                   .usage = SDL_GPU_TEXTUREUSAGE_COLOR_TARGET |
                                            SDL_GPU_TEXTUREUSAGE_SAMPLER,
                                   .width = VX_SKY_LUT_WIDTH,
                                   .height = VX_SKY_LUT_HEIGHT,
                                   .layer_count_or_depth = 1,
                                   .num_levels = 1,
                                   .sample_count = SDL_GPU_SAMPLECOUNT_1},
        "sky-view-lut");
    if (!vxray_instance.sky_view_texture)
    {
        SDL_LogError(SDL_LOG_CATEGORY_GPU, "Couldn't create sky-view LUT: %s", SDL_GetError());
        return SDL_APP_FAILURE;
    }

    // Fullscreen display pipeline

    {
        SDL_GPUShaderCreateInfo const vs_info = {.code_size = FULLSCREEN_VS_SIZE,
                                                 .code = FULLSCREEN_VS_BYTES,
                                                 .entrypoint = GPU_SHADER_ENTRYPOINT,
                                                 .format = GPU_SHADER_FORMAT,
                                                 .stage = SDL_GPU_SHADERSTAGE_VERTEX};
        SDL_GPUShaderCreateInfo const ps_info = {.code_size = DISPLAY_PS_SIZE,
                                                 .code = DISPLAY_PS_BYTES,
                                                 .entrypoint = GPU_SHADER_ENTRYPOINT,
                                                 .format = GPU_SHADER_FORMAT,
                                                 .stage = SDL_GPU_SHADERSTAGE_FRAGMENT,
                                                 .num_samplers = 3,
                                                 .num_storage_textures = 3,
                                                 .num_uniform_buffers = 1};
        SDL_GPUShader* const          vertex_shader =
            SDL_CreateGPUShader(vxray_instance.gpu_device, &vs_info);
        SDL_GPUShader* const fragment_shader =
            SDL_CreateGPUShader(vxray_instance.gpu_device, &ps_info);
        if (!vertex_shader || !fragment_shader)
        {
            SDL_LogError(SDL_LOG_CATEGORY_GPU, "Couldn't create display shaders: %s",
                         SDL_GetError());
            if (fragment_shader)
            {
                SDL_ReleaseGPUShader(vxray_instance.gpu_device, fragment_shader);
            }
            if (vertex_shader)
            {
                SDL_ReleaseGPUShader(vxray_instance.gpu_device, vertex_shader);
            }
            return SDL_APP_FAILURE;
        }

        vxray_instance.display_pipeline = vx_create_gpu_graphics_pipeline(
            vxray_instance.gpu_device,
            (SDL_GPUGraphicsPipelineCreateInfo){
                .vertex_shader = vertex_shader,
                .fragment_shader = fragment_shader,
                .primitive_type = SDL_GPU_PRIMITIVETYPE_TRIANGLELIST,
                .rasterizer_state =
                    (SDL_GPURasterizerState){.fill_mode = SDL_GPU_FILLMODE_FILL,
                                             .cull_mode = SDL_GPU_CULLMODE_BACK,
                                             .front_face = SDL_GPU_FRONTFACE_CLOCKWISE},
                .target_info =
                    (SDL_GPUGraphicsPipelineTargetInfo){
                        .num_color_targets = 1,
                        .color_target_descriptions =
                            (SDL_GPUColorTargetDescription[]){
                                {.format = SDL_GetGPUSwapchainTextureFormat(
                                     vxray_instance.gpu_device, vxray_instance.window)}}}},
            "display-raster");
        SDL_ReleaseGPUShader(vxray_instance.gpu_device, fragment_shader);
        SDL_ReleaseGPUShader(vxray_instance.gpu_device, vertex_shader);
        if (!vxray_instance.display_pipeline)
        {
            SDL_LogError(SDL_LOG_CATEGORY_GPU, "Couldn't create display pipeline: %s",
                         SDL_GetError());
            return SDL_APP_FAILURE;
        }
    }

    vxray_instance.display_sampler = SDL_CreateGPUSampler(
        vxray_instance.gpu_device,
        &(SDL_GPUSamplerCreateInfo){.min_filter = SDL_GPU_FILTER_NEAREST,
                                    .mag_filter = SDL_GPU_FILTER_NEAREST,
                                    .mipmap_mode = SDL_GPU_SAMPLERMIPMAPMODE_NEAREST,
                                    .address_mode_u = SDL_GPU_SAMPLERADDRESSMODE_CLAMP_TO_EDGE,
                                    .address_mode_v = SDL_GPU_SAMPLERADDRESSMODE_CLAMP_TO_EDGE,
                                    .address_mode_w = SDL_GPU_SAMPLERADDRESSMODE_CLAMP_TO_EDGE});
    if (!vxray_instance.display_sampler)
    {
        SDL_LogError(SDL_LOG_CATEGORY_GPU, "Couldn't create display sampler: %s", SDL_GetError());
        return SDL_APP_FAILURE;
    }

    vxray_instance.sky_view_sampler = SDL_CreateGPUSampler(
        vxray_instance.gpu_device,
        &(SDL_GPUSamplerCreateInfo){.min_filter = SDL_GPU_FILTER_LINEAR,
                                    .mag_filter = SDL_GPU_FILTER_LINEAR,
                                    .mipmap_mode = SDL_GPU_SAMPLERMIPMAPMODE_NEAREST,
                                    .address_mode_u = SDL_GPU_SAMPLERADDRESSMODE_CLAMP_TO_EDGE,
                                    .address_mode_v = SDL_GPU_SAMPLERADDRESSMODE_CLAMP_TO_EDGE,
                                    .address_mode_w = SDL_GPU_SAMPLERADDRESSMODE_CLAMP_TO_EDGE});
    if (!vxray_instance.sky_view_sampler)
    {
        SDL_LogError(SDL_LOG_CATEGORY_GPU, "Couldn't create sky-view sampler: %s", SDL_GetError());
        return SDL_APP_FAILURE;
    }

    {
        vx_scene scene = {0};
        {
            char const* vox_file = argv[1];

            SDL_PathInfo info;
            if (!SDL_GetPathInfo(vox_file, &info))
            {
                SDL_LogError(SDL_LOG_CATEGORY_APPLICATION, "%s doesn't exist", vox_file);
                return SDL_APP_FAILURE;
            }
            if (!vx_load_scene(vox_file, &scene))
            {
                return SDL_APP_FAILURE;
            }
            assert(scene.voxel_grid.ptr);
            assert(scene.voxel_masks.ptr);
            assert(scene.brick_masks.ptr);
            assert(scene.chunk_masks.ptr);
            assert(scene.grid_ext);
            vxray_instance.grid_ext = scene.grid_ext;
        }

        {
            SDL_GPUDevice* const device = vxray_instance.gpu_device;

            if (!SDL_GPUTextureSupportsFormat(vxray_instance.gpu_device,
                                              SDL_GPU_TEXTUREFORMAT_R8_UINT, SDL_GPU_TEXTURETYPE_3D,
                                              SDL_GPU_TEXTUREUSAGE_GRAPHICS_STORAGE_READ |
                                                  SDL_GPU_TEXTUREUSAGE_COMPUTE_STORAGE_READ))
            {
                SDL_LogError(SDL_LOG_CATEGORY_GPU,
                             "TEXTUREFORMAT_R8_UINT not supported on this device");
                vx_scene_free(&scene);
                return SDL_APP_FAILURE;
            }
            if (!SDL_GPUTextureSupportsFormat(vxray_instance.gpu_device,
                                              SDL_GPU_TEXTUREFORMAT_R32_UINT,
                                              SDL_GPU_TEXTURETYPE_3D,
                                              SDL_GPU_TEXTUREUSAGE_GRAPHICS_STORAGE_READ |
                                                  SDL_GPU_TEXTUREUSAGE_COMPUTE_STORAGE_READ |
                                                  SDL_GPU_TEXTUREUSAGE_COMPUTE_STORAGE_WRITE))
            {
                SDL_LogError(SDL_LOG_CATEGORY_GPU,
                             "TEXTUREFORMAT_R32_UINT not supported on this device");
                vx_scene_free(&scene);
                return SDL_APP_FAILURE;
            }

            {
                uint32_t const        voxel_buffer_size = (uint32_t)scene.voxel_grid.count;
                SDL_GPUTexture* const voxel_texture = vx_create_gpu_texture(
                    device,
                    (SDL_GPUTextureCreateInfo){.type = SDL_GPU_TEXTURETYPE_3D,
                                               .format = SDL_GPU_TEXTUREFORMAT_R8_UINT,
                                               .usage = SDL_GPU_TEXTUREUSAGE_GRAPHICS_STORAGE_READ,
                                               .width = scene.grid_ext,
                                               .height = scene.grid_ext,
                                               .layer_count_or_depth = scene.grid_ext,
                                               .num_levels = 1,
                                               .sample_count = SDL_GPU_SAMPLECOUNT_1},
                    "voxels");
                if (!voxel_texture)
                {
                    SDL_LogError(SDL_LOG_CATEGORY_GPU, "Failed to create voxel texture: %s",
                                 SDL_GetError());
                    vx_scene_free(&scene);
                    return SDL_APP_FAILURE;
                }
                if (!vx_gpu_texture_upload(device, voxel_texture, scene.voxel_grid.ptr,
                                           voxel_buffer_size, (uint32_t)scene.grid_ext))
                {
                    vx_scene_free(&scene);
                    return SDL_APP_FAILURE;
                }
                vxray_instance.voxel_texture = voxel_texture;
            }
            {
                SDL_GPUTexture* const voxel_mask_texture = vx_create_occupancy_mask_texture(
                    device, scene.voxel_masks, scene.grid_ext, "voxel-occupancy-masks");
                if (!voxel_mask_texture)
                {
                    vx_scene_free(&scene);
                    return SDL_APP_FAILURE;
                }
                vxray_instance.voxel_mask_texture = voxel_mask_texture;
            }
            {
                SDL_GPUTexture* const brick_mask_texture = vx_create_occupancy_mask_texture(
                    device, scene.brick_masks, scene.brick_grid_ext, "brick-occupancy-masks");
                if (!brick_mask_texture)
                {
                    vx_scene_free(&scene);
                    return SDL_APP_FAILURE;
                }
                vxray_instance.brick_mask_texture = brick_mask_texture;
            }
            {
                SDL_GPUTexture* const chunk_mask_texture = vx_create_occupancy_mask_texture(
                    device, scene.chunk_masks, scene.chunk_grid_ext, "chunk-occupancy-masks");
                if (!chunk_mask_texture)
                {
                    vx_scene_free(&scene);
                    return SDL_APP_FAILURE;
                }
                vxray_instance.chunk_mask_texture = chunk_mask_texture;
            }
            if (!vx_create_gpu_aadfs(
                    device, vxray_instance.voxel_mask_texture, vxray_instance.brick_mask_texture,
                    vxray_instance.chunk_mask_texture, (uint32_t)scene.grid_ext,
                    (uint32_t)scene.brick_grid_ext, (uint32_t)scene.chunk_grid_ext,
                    &vxray_instance.voxel_aadf_texture, &vxray_instance.brick_aadf_texture,
                    &vxray_instance.chunk_aadf_texture))
            {
                vx_scene_free(&scene);
                return SDL_APP_FAILURE;
            }
            {
                uint32_t const palette_size = 4 * 256;
                assert(palette_size == sizeof(scene.palette));
                SDL_GPUBuffer* const palette_buffer = vx_create_gpu_buffer(
                    device,
                    (SDL_GPUBufferCreateInfo){.usage = SDL_GPU_BUFFERUSAGE_GRAPHICS_STORAGE_READ,
                                              .size = palette_size},
                    "palette");
                if (!palette_buffer)
                {
                    SDL_LogError(SDL_LOG_CATEGORY_GPU, "Failed to create palette buffer: %s",
                                 SDL_GetError());
                    vx_scene_free(&scene);
                    return SDL_APP_FAILURE;
                }
                if (!vx_gpu_buffer_upload(device, palette_buffer, scene.palette, palette_size))
                {
                    SDL_ReleaseGPUBuffer(device, palette_buffer);
                    vx_scene_free(&scene);
                    return SDL_APP_FAILURE;
                }

                vxray_instance.palette_buffer = palette_buffer;
            }
            {
                uint32_t const ao_buffer_size = (uint32_t)(VX_AO_HASH_CAPACITY * sizeof(uint32_t));
                SDL_GPUBufferCreateInfo const buffer_info = {
                    .usage = SDL_GPU_BUFFERUSAGE_COMPUTE_STORAGE_WRITE |
                             SDL_GPU_BUFFERUSAGE_GRAPHICS_STORAGE_READ,
                    .size = ao_buffer_size};
                SDL_GPUBuffer* const checksum_buffer =
                    vx_create_gpu_buffer(device, buffer_info, "ao-checksum");
                SDL_GPUBuffer* const payload_buffer =
                    vx_create_gpu_buffer(device, buffer_info, "ao-payload");
                SDL_GPUBuffer* const last_touched_frame_buffer =
                    vx_create_gpu_buffer(device, buffer_info, "ao-last-touched-frame");
                if (!checksum_buffer || !payload_buffer || !last_touched_frame_buffer)
                {
                    SDL_LogError(SDL_LOG_CATEGORY_GPU, "Failed to create AO hash buffers: %s",
                                 SDL_GetError());
                    if (last_touched_frame_buffer)
                    {
                        SDL_ReleaseGPUBuffer(device, last_touched_frame_buffer);
                    }
                    if (payload_buffer)
                    {
                        SDL_ReleaseGPUBuffer(device, payload_buffer);
                    }
                    if (checksum_buffer)
                    {
                        SDL_ReleaseGPUBuffer(device, checksum_buffer);
                    }
                    vx_scene_free(&scene);
                    return SDL_APP_FAILURE;
                }

                SDL_GPUTransferBuffer* const reset_transfer = vx_create_gpu_transfer_buffer(
                    device,
                    (SDL_GPUTransferBufferCreateInfo){.usage = SDL_GPU_TRANSFERBUFFERUSAGE_UPLOAD,
                                                      .size = ao_buffer_size},
                    "ao-reset");
                if (!reset_transfer)
                {
                    SDL_LogError(SDL_LOG_CATEGORY_GPU,
                                 "Failed to create AO reset transfer buffer: %s", SDL_GetError());
                    SDL_ReleaseGPUBuffer(device, last_touched_frame_buffer);
                    SDL_ReleaseGPUBuffer(device, payload_buffer);
                    SDL_ReleaseGPUBuffer(device, checksum_buffer);
                    vx_scene_free(&scene);
                    return SDL_APP_FAILURE;
                }
                void* const zero_data = SDL_MapGPUTransferBuffer(device, reset_transfer, false);
                if (!zero_data)
                {
                    SDL_LogError(SDL_LOG_CATEGORY_GPU, "Failed to map AO reset buffer: %s",
                                 SDL_GetError());
                    SDL_ReleaseGPUTransferBuffer(device, reset_transfer);
                    SDL_ReleaseGPUBuffer(device, last_touched_frame_buffer);
                    SDL_ReleaseGPUBuffer(device, payload_buffer);
                    SDL_ReleaseGPUBuffer(device, checksum_buffer);
                    vx_scene_free(&scene);
                    return SDL_APP_FAILURE;
                }
                SDL_memset(zero_data, 0, ao_buffer_size);
                SDL_UnmapGPUTransferBuffer(device, reset_transfer);

                vxray_instance.ao_checksum_buffer = checksum_buffer;
                vxray_instance.ao_payload_buffer = payload_buffer;
                vxray_instance.ao_last_touched_frame_buffer = last_touched_frame_buffer;
                vxray_instance.ao_reset_transfer_buffer = reset_transfer;
            }

            vx_scene_free(&scene);
        }

        if (argc == 3)
        {
            if (!vx_camera_load(&vxray_instance.camera, argv[2]))
            {
                return SDL_APP_FAILURE;
            }
        }
        else
        {
            float const view_radius = 0.5f * (float)scene.grid_ext;
            vxray_instance.camera.position =
                glms_vec3_add(scene.center, (vec3s){0.f, view_radius * 0.3f, -view_radius * 2.8f});
            vx_camera_look_at(&vxray_instance.camera, scene.center);
        }
    }

    return SDL_APP_CONTINUE;
}

SDL_AppResult SDL_AppEvent(void* const appstate, SDL_Event* const event)
{
    (void)appstate;

    imgui_sdl3_process_event(event);

    if (event->type == SDL_EVENT_QUIT)
    {
        return SDL_APP_SUCCESS;
    }

    if (!imgui_sdl3_wants_capture_keyboard())
    {
        if (event->type == SDL_EVENT_KEY_DOWN && !event->key.repeat &&
            event->key.scancode == SDL_SCANCODE_F2)
        {
            vx_camera_print_code(&vxray_instance.camera);
        }
    }

    bool const application_has_mouse_capture =
        (vxray_instance.input.pointer_events & VX_INPUT_POINTER_PRESSED) != 0;
    if (!imgui_sdl3_wants_capture_mouse() || application_has_mouse_capture)
    {
        if (event->type == SDL_EVENT_MOUSE_BUTTON_DOWN && event->button.button == SDL_BUTTON_LEFT)
        {
            vxray_instance.input.pointer_events |= VX_INPUT_POINTER_DOWN | VX_INPUT_POINTER_PRESSED;
        }
        if (event->type == SDL_EVENT_MOUSE_BUTTON_UP && event->button.button == SDL_BUTTON_LEFT)
        {
            vxray_instance.input.pointer_events |= VX_INPUT_POINTER_UP;
            vxray_instance.input.pointer_events &= ~VX_INPUT_POINTER_PRESSED;
        }
        if (event->type == SDL_EVENT_MOUSE_MOTION)
        {
            vxray_instance.input.pointer_delta[0] += (float)event->motion.xrel;
            vxray_instance.input.pointer_delta[1] += (float)event->motion.yrel;
        }
    }

    return SDL_APP_CONTINUE;
}

SDL_AppResult SDL_AppIterate(void* const appstate)
{
    (void)appstate;

    SDL_GPUDevice* const gpu_device = vxray_instance.gpu_device;
    SDL_Window* const    window = vxray_instance.window;
    vx_camera* const     camera = &vxray_instance.camera;
    vx_input* const      input = &vxray_instance.input;

    //
    // Input
    //

    if (input->pointer_events & VX_INPUT_POINTER_DOWN)
    {
        if (!SDL_SetWindowRelativeMouseMode(window, true))
        {
            SDL_LogWarn(SDL_LOG_CATEGORY_INPUT, "Couldn't enable relative mouse mode: %s",
                        SDL_GetError());
        }
    }
    if (input->pointer_events & VX_INPUT_POINTER_UP)
    {
        if (!SDL_SetWindowRelativeMouseMode(window, false))
        {
            SDL_LogWarn(SDL_LOG_CATEGORY_INPUT, "Couldn't disable relative mouse mode: %s",
                        SDL_GetError());
        }
    }
    if (input->pointer_events & VX_INPUT_POINTER_PRESSED)
    {
        float const mouse_sensitivity = 0.003f;
        float const pitch_limit = 1.55334306f;
        camera->yaw += input->pointer_delta[0] * mouse_sensitivity;
        camera->pitch = SDL_clamp(camera->pitch - input->pointer_delta[1] * mouse_sensitivity,
                                  -pitch_limit, pitch_limit);
    }

    //
    // Camera
    //

    if (!imgui_sdl3_wants_capture_keyboard())
    {
        vx_camera_update_movement(camera);
    }

    //
    // Reset input state
    //

    input->pointer_events &= VX_INPUT_POINTER_PRESSED;
    input->pointer_delta[0] = 0.f;
    input->pointer_delta[1] = 0.f;

    //
    // Render
    //

    imgui_sdl3_new_frame();
    igBegin("Display", 0, 0);
    igRadioButton_IntPtr("Shaded", &vxray_instance.display_texture, VX_DISPLAY_TEXTURE_ALBEDO);
    igRadioButton_IntPtr("Normal", &vxray_instance.display_texture, VX_DISPLAY_TEXTURE_NORMAL);
    igRadioButton_IntPtr("Surface depth", &vxray_instance.display_texture,
                         VX_DISPLAY_TEXTURE_SURFACE_DEPTH);
    igRadioButton_IntPtr("Ambient visibility", &vxray_instance.display_texture,
                         VX_DISPLAY_TEXTURE_AMBIENT_VISIBILITY);
    igRadioButton_IntPtr("Cell size", &vxray_instance.display_texture,
                         VX_DISPLAY_TEXTURE_CELL_SIZE);
    igRadioButton_IntPtr("Spatial index", &vxray_instance.display_texture,
                         VX_DISPLAY_TEXTURE_SPATIAL_INDEX);
    igRadioButton_IntPtr("Sky-view LUT", &vxray_instance.display_texture,
                         VX_DISPLAY_TEXTURE_SKY_VIEW);
    bool invalidate_ao =
        igSliderFloat("rtao_radius", &vxray_instance.rtao_radius, 8.f, 16.f, "%.1f voxels", 0);
    igSliderInt("RTAO samples/pixel/frame", &vxray_instance.rtao_samples_per_frame, 1,
                VX_RTAO_MAX_SAMPLES_PER_FRAME, "%d", 0);
    invalidate_ao |= igSliderFloat("sp", &vxray_instance.ao_sp, 1.f, 16.f, "%.1f px", 0);
    invalidate_ao |= igSliderFloat("smin", &vxray_instance.ao_smin, 0.01f, 2.f, "%.3f",
                                   ImGuiSliderFlags_Logarithmic);
    bool sky_view_changed = igSliderFloat("Sun elevation", &vxray_instance.sun_elevation_degrees,
                                          -10.f, 90.f, "%.1f deg", 0);
    sky_view_changed |= igSliderFloat("Sun azimuth", &vxray_instance.sun_azimuth_degrees, 0.f,
                                      360.f, "%.1f deg", 0);
    sky_view_changed |=
        igSliderFloat("View altitude", &vxray_instance.view_altitude_km, 0.f, 25.f, "%.1f km", 0);
    vxray_instance.sky_view_dirty |= sky_view_changed;
    igEnd();
    igRender();

    SDL_GPUCommandBuffer* const cmd_buffer = SDL_AcquireGPUCommandBuffer(gpu_device);
    if (!cmd_buffer)
    {
        SDL_LogError(SDL_LOG_CATEGORY_GPU, "Couldn't acquire GPU command buffer: %s\n",
                     SDL_GetError());
        return SDL_APP_FAILURE;
    }

    SDL_GPUTexture* swapchain_texture = 0;
    uint32_t        width, height;
    if (!SDL_WaitAndAcquireGPUSwapchainTexture(cmd_buffer, window, &swapchain_texture, &width,
                                               &height))
    {
        SDL_LogError(SDL_LOG_CATEGORY_GPU, "Couldn't acquire GPU swapchain texture: %s",
                     SDL_GetError());
        if (!SDL_CancelGPUCommandBuffer(cmd_buffer))
        {
            SDL_LogError(SDL_LOG_CATEGORY_GPU, "Couldn't cancel GPU command buffer: %s",
                         SDL_GetError());
        }
        return SDL_APP_FAILURE;
    }

    if (!swapchain_texture)
    {
        if (!SDL_SubmitGPUCommandBuffer(cmd_buffer))
        {
            SDL_LogError(SDL_LOG_CATEGORY_GPU, "Couldn't submit GPU command buffer: %s",
                         SDL_GetError());
            return SDL_APP_FAILURE;
        }
        return SDL_APP_CONTINUE;
    }

    if (!vx_ensure_render_textures(width, height))
    {
        SDL_CancelGPUCommandBuffer(cmd_buffer);
        return SDL_APP_FAILURE;
    }

    imgui_sdl3_prepare_draw_data(cmd_buffer);

    assert(width > 0u);
    assert(height > 0u);
    assert(isfinite(vxray_instance.rtao_radius) && vxray_instance.rtao_radius >= 8.f &&
           vxray_instance.rtao_radius <= 16.f);
    assert(vxray_instance.rtao_samples_per_frame >= 1 &&
           vxray_instance.rtao_samples_per_frame <= VX_RTAO_MAX_SAMPLES_PER_FRAME);
    assert(isfinite(vxray_instance.ao_sp) && vxray_instance.ao_sp > 0.f);
    assert(isfinite(vxray_instance.ao_smin) && vxray_instance.ao_smin > 0.f);
    assert(isfinite(vxray_instance.sun_elevation_degrees) &&
           vxray_instance.sun_elevation_degrees >= -10.f &&
           vxray_instance.sun_elevation_degrees <= 90.f);
    assert(isfinite(vxray_instance.sun_azimuth_degrees) &&
           vxray_instance.sun_azimuth_degrees >= 0.f &&
           vxray_instance.sun_azimuth_degrees <= 360.f);
    assert(isfinite(vxray_instance.view_altitude_km) && vxray_instance.view_altitude_km >= 0.f &&
           vxray_instance.view_altitude_km <= 25.f);

    vec3s const forward = vx_camera_forward(camera);
    vec3s const world_up = {0.f, 1.f, 0.f};
    float const aspect = (float)width / (float)height;
    float const fov = glm_rad(60.f);
    float const near_plane = 1.f;
    float const far_plane =
        glms_vec3_norm(camera->position) + sqrtf(3.f) * (float)vxray_instance.grid_ext + 1.f;
    assert(isfinite(aspect) && aspect > 0.f);
    assert(isfinite(fov) && fov > 0.f && fov < VX_PI_F);
    assert(isfinite(near_plane) && near_plane > 0.f);
    assert(isfinite(far_plane) && far_plane > near_plane);
    mat4s const            view = glms_look_lh_zo(camera->position, forward, world_up);
    mat4s                  projection = glms_perspective_lh_zo(fov, aspect, near_plane, far_plane);
    mat4s const            view_projection = glms_mat4_mul(projection, view);
    mat4s const            inverse_view_projection = glms_mat4_inv(view_projection);
    float4x4 const         view_projection_data = vx_float4x4_from_mat4(view_projection);
    gbuffer_uniforms const gbuffer_uniform_data = {
        .camera_pos = vx_float4_from_vec3(camera->position, 0.f),
        .inverse_view_projection = vx_float4x4_from_mat4(inverse_view_projection),
        .view_projection = view_projection_data,
        .grid_ext = vxray_instance.grid_ext};
    bool const    reset_ao = vxray_instance.frame_index == 0u || invalidate_ao;
    rtao_uniforms rtao_uniform_data = {
        .camera_pos = vx_float4_from_vec3(camera->position, 0.f),
        .inverse_view_projection = vx_float4x4_from_mat4(inverse_view_projection),
        .previous_view_projection = vxray_instance.previous_view_projection,
        .rtao_radius = vxray_instance.rtao_radius,
        .sp = vxray_instance.ao_sp,
        .smin = vxray_instance.ao_smin,
        .vertical_fov = fov,
        .near_plane = near_plane,
        .far_plane = far_plane,
        .grid_ext = vxray_instance.grid_ext,
        .frame_index = vxray_instance.frame_index,
        .render_height = height,
        .sample_index = 0u,
        .history_valid = (uint)(vxray_instance.rtao_history_valid && !reset_ao)};
    float const             sun_elevation = glm_rad(vxray_instance.sun_elevation_degrees);
    float const             sun_azimuth = glm_rad(vxray_instance.sun_azimuth_degrees);
    float const             cos_sun_elevation = cosf(sun_elevation);
    sky_view_uniforms const sky_view_uniform_data = {
        .view_position = float4(0.f, 1000.f * vxray_instance.view_altitude_km, 0.f, 0.f),
        .sun_direction = float4(cos_sun_elevation * sinf(sun_azimuth), sinf(sun_elevation),
                                cos_sun_elevation * cosf(sun_azimuth), 0.f),
        .sun_color = VX_SKY_SUN_COLOR};

    if (reset_ao)
    {
        SDL_GPUCopyPass* const copy_pass = SDL_BeginGPUCopyPass(cmd_buffer);
        assert(copy_pass);
        uint32_t const ao_buffer_size = (uint32_t)(VX_AO_HASH_CAPACITY * sizeof(uint32_t));
        SDL_GPUTransferBufferLocation const reset_source = {
            .transfer_buffer = vxray_instance.ao_reset_transfer_buffer, .offset = 0};
        SDL_UploadToGPUBuffer(copy_pass, &reset_source,
                              &(SDL_GPUBufferRegion){.buffer = vxray_instance.ao_checksum_buffer,
                                                     .offset = 0,
                                                     .size = ao_buffer_size},
                              false);
        SDL_UploadToGPUBuffer(copy_pass, &reset_source,
                              &(SDL_GPUBufferRegion){.buffer = vxray_instance.ao_payload_buffer,
                                                     .offset = 0,
                                                     .size = ao_buffer_size},
                              false);
        SDL_UploadToGPUBuffer(
            copy_pass, &reset_source,
            &(SDL_GPUBufferRegion){.buffer = vxray_instance.ao_last_touched_frame_buffer,
                                   .offset = 0,
                                   .size = ao_buffer_size},
            false);
        SDL_EndGPUCopyPass(copy_pass);
    }

    // Regenerate the sky-view LUT when its atmospheric view parameters change.

    if (vxray_instance.sky_view_dirty)
    {
        SDL_GPUColorTargetInfo const sky_view_target_info = {
            .texture = vxray_instance.sky_view_texture,
            .clear_color = (SDL_FColor){0.f, 0.f, 0.f, 0.f},
            .load_op = SDL_GPU_LOADOP_CLEAR,
            .store_op = SDL_GPU_STOREOP_STORE,
            .cycle = true};
        SDL_GPURenderPass* const sky_view_pass =
            SDL_BeginGPURenderPass(cmd_buffer, &sky_view_target_info, 1, 0);
        assert(sky_view_pass);
        SDL_BindGPUGraphicsPipeline(sky_view_pass, vxray_instance.sky_view_pipeline);
        SDL_PushGPUFragmentUniformData(cmd_buffer, 0, &sky_view_uniform_data,
                                       sizeof(sky_view_uniform_data));
        SDL_DrawGPUPrimitives(sky_view_pass, 3, 1, 0, 0);
        SDL_EndGPURenderPass(sky_view_pass);
        vxray_instance.sky_view_dirty = false;
    }

    // Trace the primary voxel rays into the G-buffer.

    SDL_GPUColorTargetInfo const gbuffer_target_info[] = {
        {.texture = vxray_instance.gbuffer_albedo_texture,
         .clear_color = (SDL_FColor){0.f, 0.f, 0.f, 0.f},
         .load_op = SDL_GPU_LOADOP_CLEAR,
         .store_op = SDL_GPU_STOREOP_STORE,
         .cycle = true},
        {.texture = vxray_instance.gbuffer_normal_texture,
         .clear_color = (SDL_FColor){0.f, 0.f, 0.f, 0.f},
         .load_op = SDL_GPU_LOADOP_CLEAR,
         .store_op = SDL_GPU_STOREOP_STORE,
         .cycle = true}};
    SDL_GPUDepthStencilTargetInfo const gbuffer_depth_target_info = {
        .texture = vxray_instance.gbuffer_depth_texture,
        .clear_depth = 1.f,
        .load_op = SDL_GPU_LOADOP_CLEAR,
        .store_op = SDL_GPU_STOREOP_STORE,
        .stencil_load_op = SDL_GPU_LOADOP_DONT_CARE,
        .stencil_store_op = SDL_GPU_STOREOP_DONT_CARE,
        .cycle = true};
    SDL_GPURenderPass* const gbuffer_pass =
        SDL_BeginGPURenderPass(cmd_buffer, gbuffer_target_info, SDL_arraysize(gbuffer_target_info),
                               &gbuffer_depth_target_info);
    assert(gbuffer_pass);
    SDL_BindGPUGraphicsPipeline(gbuffer_pass, vxray_instance.gbuffer_pipeline);
    SDL_GPUTexture* const storage_textures[] = {
        vxray_instance.voxel_texture,      vxray_instance.voxel_mask_texture,
        vxray_instance.brick_mask_texture, vxray_instance.chunk_mask_texture,
        vxray_instance.voxel_aadf_texture, vxray_instance.brick_aadf_texture,
        vxray_instance.chunk_aadf_texture,
    };
    SDL_BindGPUFragmentStorageTextures(gbuffer_pass, 0, storage_textures,
                                       SDL_arraysize(storage_textures));
    SDL_GPUBuffer* const storage_buffers[] = {vxray_instance.palette_buffer};
    SDL_BindGPUFragmentStorageBuffers(gbuffer_pass, 0, storage_buffers,
                                      SDL_arraysize(storage_buffers));
    SDL_PushGPUFragmentUniformData(cmd_buffer, 0, &gbuffer_uniform_data,
                                   sizeof(gbuffer_uniform_data));
    SDL_DrawGPUPrimitives(gbuffer_pass, 3, 1, 0, 0);
    SDL_EndGPURenderPass(gbuffer_pass);

    // Reproject the previous spatial-cache index, falling back to hash lookup on a miss.

    uint32_t const               current_history = vxray_instance.frame_index & 1u;
    uint32_t const               previous_history = current_history ^ 1u;
    SDL_GPUColorTargetInfo const rtao_index_target_info[] = {
        {.texture = vxray_instance.rtao_index_textures[current_history],
         .load_op = SDL_GPU_LOADOP_DONT_CARE,
         .store_op = SDL_GPU_STOREOP_STORE,
         .cycle = true},
        {.texture = vxray_instance.rtao_checksum_textures[current_history],
         .load_op = SDL_GPU_LOADOP_DONT_CARE,
         .store_op = SDL_GPU_STOREOP_STORE,
         .cycle = true}};
    SDL_GPURenderPass* const rtao_index_pass = SDL_BeginGPURenderPass(
        cmd_buffer, rtao_index_target_info, SDL_arraysize(rtao_index_target_info), 0);
    assert(rtao_index_pass);
    SDL_BindGPUGraphicsPipeline(rtao_index_pass, vxray_instance.rtao_index_pipeline);
    SDL_GPUTextureSamplerBinding const rtao_index_depth_binding = {
        .texture = vxray_instance.gbuffer_depth_texture, .sampler = vxray_instance.display_sampler};
    SDL_BindGPUFragmentSamplers(rtao_index_pass, 0, &rtao_index_depth_binding, 1);
    SDL_GPUTexture* const rtao_index_storage_textures[] = {
        vxray_instance.gbuffer_normal_texture,
        vxray_instance.rtao_index_textures[previous_history],
        vxray_instance.rtao_checksum_textures[previous_history],
    };
    SDL_BindGPUFragmentStorageTextures(rtao_index_pass, 0, rtao_index_storage_textures,
                                       SDL_arraysize(rtao_index_storage_textures));
    SDL_GPUBuffer* const rtao_index_storage_buffers[] = {
        vxray_instance.ao_checksum_buffer,
        vxray_instance.ao_payload_buffer,
        vxray_instance.ao_last_touched_frame_buffer,
    };
    SDL_BindGPUFragmentStorageBuffers(rtao_index_pass, 0, rtao_index_storage_buffers,
                                      SDL_arraysize(rtao_index_storage_buffers));
    SDL_PushGPUFragmentUniformData(cmd_buffer, 0, &rtao_uniform_data, sizeof(rtao_uniform_data));
    SDL_DrawGPUPrimitives(rtao_index_pass, 3, 1, 0, 0);
    SDL_EndGPURenderPass(rtao_index_pass);

    // Add one RTAO sample per pixel and pass, resolving visibility after every sample.

    SDL_GPUTextureSamplerBinding const rtao_depth_binding = {
        .texture = vxray_instance.gbuffer_depth_texture, .sampler = vxray_instance.display_sampler};
    SDL_GPUTexture* const rtao_storage_textures[] = {
        vxray_instance.rtao_index_textures[current_history],
        vxray_instance.gbuffer_normal_texture,
        vxray_instance.voxel_mask_texture,
        vxray_instance.voxel_aadf_texture,
    };
    SDL_GPUBuffer* const rtao_storage_buffers[] = {
        vxray_instance.ao_payload_buffer,
    };
    for (int sample_index = 0; sample_index < vxray_instance.rtao_samples_per_frame; ++sample_index)
    {
        rtao_uniform_data.sample_index = (uint)sample_index;
        SDL_GPUColorTargetInfo const rtao_target_info = {
            .texture = vxray_instance.rtao_visibility_texture,
            .load_op = SDL_GPU_LOADOP_DONT_CARE,
            .store_op = sample_index == vxray_instance.rtao_samples_per_frame - 1
                            ? SDL_GPU_STOREOP_STORE
                            : SDL_GPU_STOREOP_DONT_CARE,
            .cycle = true};
        SDL_GPURenderPass* const rtao_pass =
            SDL_BeginGPURenderPass(cmd_buffer, &rtao_target_info, 1, 0);
        assert(rtao_pass);
        SDL_BindGPUGraphicsPipeline(rtao_pass, vxray_instance.rtao_pipeline);
        SDL_BindGPUFragmentSamplers(rtao_pass, 0, &rtao_depth_binding, 1);
        SDL_BindGPUFragmentStorageTextures(rtao_pass, 0, rtao_storage_textures,
                                           SDL_arraysize(rtao_storage_textures));
        SDL_BindGPUFragmentStorageBuffers(rtao_pass, 0, rtao_storage_buffers,
                                          SDL_arraysize(rtao_storage_buffers));
        SDL_PushGPUFragmentUniformData(cmd_buffer, 0, &rtao_uniform_data,
                                       sizeof(rtao_uniform_data));
        SDL_DrawGPUPrimitives(rtao_pass, 3, 1, 0, 0);
        SDL_EndGPURenderPass(rtao_pass);
    }

    // Display the selected intermediate texture on the swapchain.

    SDL_GPUColorTargetInfo const color_target_info = {.texture = swapchain_texture,
                                                      .clear_color =
                                                          (SDL_FColor){0.f, 0.f, 0.f, 0.f},
                                                      .load_op = SDL_GPU_LOADOP_CLEAR,
                                                      .store_op = SDL_GPU_STOREOP_STORE};
    SDL_GPURenderPass* render_pass = SDL_BeginGPURenderPass(cmd_buffer, &color_target_info, 1, 0);
    assert(render_pass);

    SDL_BindGPUGraphicsPipeline(render_pass, vxray_instance.display_pipeline);
    SDL_GPUTextureSamplerBinding const display_bindings[] = {
        {.texture = vxray_instance.gbuffer_depth_texture,
         .sampler = vxray_instance.display_sampler},
        {.texture = vxray_instance.rtao_visibility_texture,
         .sampler = vxray_instance.display_sampler},
        {.texture = vxray_instance.sky_view_texture, .sampler = vxray_instance.sky_view_sampler},
    };
    SDL_BindGPUFragmentSamplers(render_pass, 0, display_bindings, SDL_arraysize(display_bindings));
    SDL_GPUTexture* const display_storage_textures[] = {
        vxray_instance.gbuffer_albedo_texture,
        vxray_instance.gbuffer_normal_texture,
        vxray_instance.rtao_index_textures[current_history],
    };
    SDL_BindGPUFragmentStorageTextures(render_pass, 0, display_storage_textures,
                                       SDL_arraysize(display_storage_textures));
    display_uniforms const display_uniform_data = {.texture_type =
                                                       (uint)vxray_instance.display_texture,
                                                   .near_plane = near_plane,
                                                   .far_plane = far_plane,
                                                   .grid_ext = (uint)vxray_instance.grid_ext,
                                                   .sp = vxray_instance.ao_sp,
                                                   .smin = vxray_instance.ao_smin,
                                                   .vertical_fov = fov,
                                                   .render_height = height};
    SDL_PushGPUFragmentUniformData(cmd_buffer, 0, &display_uniform_data,
                                   sizeof(display_uniform_data));
    SDL_DrawGPUPrimitives(render_pass, 3, 1, 0, 0);
    imgui_sdl3_render_draw_data(cmd_buffer, render_pass);
    SDL_EndGPURenderPass(render_pass);
    if (!SDL_SubmitGPUCommandBuffer(cmd_buffer))
    {
        SDL_LogError(SDL_LOG_CATEGORY_GPU, "Couldn't submit GPU command buffer: %s",
                     SDL_GetError());
        return SDL_APP_FAILURE;
    }

    vxray_instance.previous_view_projection = view_projection_data;
    vxray_instance.rtao_history_valid = true;
    ++vxray_instance.frame_index;

    return SDL_APP_CONTINUE;
}

void SDL_AppQuit(void* const appstate, SDL_AppResult const result)
{
    (void)appstate;
    (void)result;

    imgui_sdl3_shutdown();

    if (vxray_instance.ao_reset_transfer_buffer)
    {
        SDL_ReleaseGPUTransferBuffer(vxray_instance.gpu_device,
                                     vxray_instance.ao_reset_transfer_buffer);
        vxray_instance.ao_reset_transfer_buffer = 0;
    }

    if (vxray_instance.ao_payload_buffer)
    {
        SDL_ReleaseGPUBuffer(vxray_instance.gpu_device, vxray_instance.ao_payload_buffer);
        vxray_instance.ao_payload_buffer = 0;
    }

    if (vxray_instance.ao_last_touched_frame_buffer)
    {
        SDL_ReleaseGPUBuffer(vxray_instance.gpu_device,
                             vxray_instance.ao_last_touched_frame_buffer);
        vxray_instance.ao_last_touched_frame_buffer = 0;
    }

    if (vxray_instance.ao_checksum_buffer)
    {
        SDL_ReleaseGPUBuffer(vxray_instance.gpu_device, vxray_instance.ao_checksum_buffer);
        vxray_instance.ao_checksum_buffer = 0;
    }

    if (vxray_instance.palette_buffer)
    {
        SDL_ReleaseGPUBuffer(vxray_instance.gpu_device, vxray_instance.palette_buffer);
        vxray_instance.palette_buffer = 0;
    }

    if (vxray_instance.voxel_texture)
    {
        SDL_ReleaseGPUTexture(vxray_instance.gpu_device, vxray_instance.voxel_texture);
        vxray_instance.voxel_texture = 0;
    }

    if (vxray_instance.brick_mask_texture)
    {
        SDL_ReleaseGPUTexture(vxray_instance.gpu_device, vxray_instance.brick_mask_texture);
        vxray_instance.brick_mask_texture = 0;
    }

    if (vxray_instance.chunk_mask_texture)
    {
        SDL_ReleaseGPUTexture(vxray_instance.gpu_device, vxray_instance.chunk_mask_texture);
        vxray_instance.chunk_mask_texture = 0;
    }

    if (vxray_instance.voxel_mask_texture)
    {
        SDL_ReleaseGPUTexture(vxray_instance.gpu_device, vxray_instance.voxel_mask_texture);
        vxray_instance.voxel_mask_texture = 0;
    }

    if (vxray_instance.chunk_aadf_texture)
    {
        SDL_ReleaseGPUTexture(vxray_instance.gpu_device, vxray_instance.chunk_aadf_texture);
        vxray_instance.chunk_aadf_texture = 0;
    }

    if (vxray_instance.brick_aadf_texture)
    {
        SDL_ReleaseGPUTexture(vxray_instance.gpu_device, vxray_instance.brick_aadf_texture);
        vxray_instance.brick_aadf_texture = 0;
    }

    if (vxray_instance.voxel_aadf_texture)
    {
        SDL_ReleaseGPUTexture(vxray_instance.gpu_device, vxray_instance.voxel_aadf_texture);
        vxray_instance.voxel_aadf_texture = 0;
    }

    if (vxray_instance.gbuffer_depth_texture)
    {
        SDL_ReleaseGPUTexture(vxray_instance.gpu_device, vxray_instance.gbuffer_depth_texture);
        vxray_instance.gbuffer_depth_texture = 0;
    }

    for (uint32_t i = 0u; i < 2u; ++i)
    {
        if (vxray_instance.rtao_checksum_textures[i])
        {
            SDL_ReleaseGPUTexture(vxray_instance.gpu_device,
                                  vxray_instance.rtao_checksum_textures[i]);
            vxray_instance.rtao_checksum_textures[i] = 0;
        }
        if (vxray_instance.rtao_index_textures[i])
        {
            SDL_ReleaseGPUTexture(vxray_instance.gpu_device, vxray_instance.rtao_index_textures[i]);
            vxray_instance.rtao_index_textures[i] = 0;
        }
    }

    if (vxray_instance.rtao_visibility_texture)
    {
        SDL_ReleaseGPUTexture(vxray_instance.gpu_device, vxray_instance.rtao_visibility_texture);
        vxray_instance.rtao_visibility_texture = 0;
    }

    if (vxray_instance.sky_view_texture)
    {
        SDL_ReleaseGPUTexture(vxray_instance.gpu_device, vxray_instance.sky_view_texture);
        vxray_instance.sky_view_texture = 0;
    }

    if (vxray_instance.gbuffer_normal_texture)
    {
        SDL_ReleaseGPUTexture(vxray_instance.gpu_device, vxray_instance.gbuffer_normal_texture);
        vxray_instance.gbuffer_normal_texture = 0;
    }

    if (vxray_instance.gbuffer_albedo_texture)
    {
        SDL_ReleaseGPUTexture(vxray_instance.gpu_device, vxray_instance.gbuffer_albedo_texture);
        vxray_instance.gbuffer_albedo_texture = 0;
    }

    if (vxray_instance.display_sampler)
    {
        SDL_ReleaseGPUSampler(vxray_instance.gpu_device, vxray_instance.display_sampler);
        vxray_instance.display_sampler = 0;
    }

    if (vxray_instance.sky_view_sampler)
    {
        SDL_ReleaseGPUSampler(vxray_instance.gpu_device, vxray_instance.sky_view_sampler);
        vxray_instance.sky_view_sampler = 0;
    }

    if (vxray_instance.gbuffer_pipeline)
    {
        SDL_ReleaseGPUGraphicsPipeline(vxray_instance.gpu_device, vxray_instance.gbuffer_pipeline);
        vxray_instance.gbuffer_pipeline = 0;
    }

    if (vxray_instance.rtao_index_pipeline)
    {
        SDL_ReleaseGPUGraphicsPipeline(vxray_instance.gpu_device,
                                       vxray_instance.rtao_index_pipeline);
        vxray_instance.rtao_index_pipeline = 0;
    }

    if (vxray_instance.rtao_pipeline)
    {
        SDL_ReleaseGPUGraphicsPipeline(vxray_instance.gpu_device, vxray_instance.rtao_pipeline);
        vxray_instance.rtao_pipeline = 0;
    }

    if (vxray_instance.sky_view_pipeline)
    {
        SDL_ReleaseGPUGraphicsPipeline(vxray_instance.gpu_device, vxray_instance.sky_view_pipeline);
        vxray_instance.sky_view_pipeline = 0;
    }

    if (vxray_instance.display_pipeline)
    {
        SDL_ReleaseGPUGraphicsPipeline(vxray_instance.gpu_device, vxray_instance.display_pipeline);
        vxray_instance.display_pipeline = 0;
    }

    if (vxray_instance.window_claimed)
    {
        assert(vxray_instance.gpu_device);
        assert(vxray_instance.window);
        SDL_ReleaseWindowFromGPUDevice(vxray_instance.gpu_device, vxray_instance.window);
    }

    if (vxray_instance.gpu_device)
    {
        SDL_DestroyGPUDevice(vxray_instance.gpu_device);
        vxray_instance.gpu_device = 0;
    }

    if (vxray_instance.window)
    {
        SDL_DestroyWindow(vxray_instance.window);
        vxray_instance.window = 0;
    }
}
