#include "constants.h"
#include "cvox.h"
#include "dda.h"
#include "hlsl_shim.h"

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
#include <SDL3/SDL_scancode.h>
#include <SDL3/SDL_stdinc.h>
#include <SDL3/SDL_video.h>

#include <assert.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

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

static float3 vx_transform_point(cvox_transform const* const t, float3 const p)
{
    float3 const q = {.x = t->m30 + (t->m00 * p.x) + (t->m10 * p.y) + (t->m20 * p.z),
                      .y = t->m31 + (t->m01 * p.x) + (t->m11 * p.y) + (t->m21 * p.z),
                      .z = t->m32 + (t->m02 * p.x) + (t->m12 * p.y) + (t->m22 * p.z)};
    // Magicavoxel seems to use z-up. Flip to y-up.
    return float3(q.x, q.z, q.y);
}

static int vx_round_to_int(float const val)
{
    return (int)(val >= 0.f ? (val + 0.5f) : (val - 0.5f));
}

static int vx_grid_index(int const x, int const y, int const z, int const grid_ext)
{
    return x + (y * grid_ext) + (z * grid_ext * grid_ext);
}

static float3 vx_float3_add(float3 const a, float3 const b)
{
    return float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

static float3 vx_float3_sub(float3 const a, float3 const b)
{
    return float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

static float3 vx_float3_scale(float3 const v, float const s)
{
    return float3(v.x * s, v.y * s, v.z * s);
}

static float vx_float3_dot(float3 const a, float3 const b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

static float3 vx_float3_cross(float3 const a, float3 const b)
{
    return float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}

static float vx_float3_norm(float3 const v) { return sqrtf(vx_float3_dot(v, v)); }

static float3 vx_float3_normalize(float3 const v)
{
    float const norm = vx_float3_norm(v);
    if (norm <= 0.f)
    {
        return float3(0.f, 0.f, 0.f);
    }

    return vx_float3_scale(v, 1.f / norm);
}

static float4 vx_float4_from_float3(float3 const v, float const w)
{
    return float4(v.x, v.y, v.z, w);
}

typedef struct vx_scene
{
    vx_buffer(uint8_t) voxel_grid;
    vx_buffer(uint8_t) macro_grid;
    uint   palette[256];
    int    grid_ext;
    int    macro_grid_ext;
    float3 center;
} vx_scene;

// Loads a MagicaVoxel scene into dense voxel and macro grids. Free both grids with
// `vx_buffer_free`.
static bool vx_load_scene(char const* const vox_path, vx_scene* const out_scene)
{
    assert(vox_path);
    assert(out_scene);

    vx_buffer(uint8_t) voxel_grid = {0};
    vx_buffer(uint8_t) macro_grid = {0};
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

            cvox_transform const transform =
                cvox_sample_instance_transform_global(instance, 0, scene);
            int3 const   pivot = {.x = (int)(model->size_x / 2),
                                  .y = (int)(model->size_y / 2),
                                  .z = (int)(model->size_z / 2)};
            float const  min_x = (float)-pivot.x;
            float const  min_y = (float)-pivot.y;
            float const  min_z = (float)-pivot.z;
            float const  max_x = (float)((int)model->size_x - 1 - pivot.x);
            float const  max_y = (float)((int)model->size_y - 1 - pivot.y);
            float const  max_z = (float)((int)model->size_z - 1 - pivot.z);
            float3 const corners[8] = {{min_x, min_y, min_z}, {max_x, min_y, min_z},
                                       {min_x, max_y, min_z}, {min_x, min_y, max_z},
                                       {max_x, max_y, min_z}, {max_x, min_y, max_z},
                                       {min_x, max_y, max_z}, {max_x, max_y, max_z}};
            for (int c = 0; c < 8; ++c)
            {
                float3 const tc = vx_transform_point(&transform, corners[c]);
                int const    rx = vx_round_to_int(tc.x);
                int const    ry = vx_round_to_int(tc.y);
                int const    rz = vx_round_to_int(tc.z);
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
        if (largest_extent > 1024) // NOTE: 1024^ 3 is (1 << 30)
        {
            SDL_LogError(SDL_LOG_CATEGORY_APPLICATION, "Scene extent is too large: %d",
                         largest_extent);
            goto cleanup_scene;
        }

        int const grid_ext = (int)vx_next_power_of_2((uint32_t)largest_extent);
        int const total_voxels = grid_ext * grid_ext * grid_ext;
        assert(total_voxels % 4 == 0);
        voxel_grid = vx_buffer_calloc(uint8_t, total_voxels);
        if (!voxel_grid.ptr)
        {
            SDL_LogError(SDL_LOG_CATEGORY_APPLICATION, "Failed to allocate voxel grid");
            goto cleanup_scene;
        }

        int const macro_grid_ext = (grid_ext + VX_MACRO_CELL_EXT - 1) / VX_MACRO_CELL_EXT;
        int const total_macro_cells = macro_grid_ext * macro_grid_ext * macro_grid_ext;
        macro_grid = vx_buffer_calloc(uint8_t, total_macro_cells);
        if (!macro_grid.ptr)
        {
            SDL_LogError(SDL_LOG_CATEGORY_APPLICATION, "Failed to allocate macro grid");
            goto cleanup_grids;
        }

        for (int i = 0; i < (int)scene->num_instances; ++i)
        {
            cvox_instance const* const instance = &scene->instances[i];
            int const                  model_idx = (int)cvox_sample_instance_model(instance, 0);

            cvox_model const* const model = scene->models[model_idx];
            assert(model);

            cvox_transform const transform =
                cvox_sample_instance_transform_global(instance, 0, scene);

            int const  sx = (int)model->size_x;
            int const  sy = (int)model->size_y;
            int const  sz = (int)model->size_z;
            int3 const pivot = {.x = (int)(model->size_x / 2),
                                .y = (int)(model->size_y / 2),
                                .z = (int)(model->size_z / 2)};
            for (int z = 0; z < sz; ++z)
            {
                for (int y = 0; y < sy; ++y)
                {
                    for (int x = 0; x < sx; ++x)
                    {
                        int const     src_idx = x + y * sx + z * sx * sy;
                        uint8_t const voxel = model->voxel_data[src_idx];
                        if (voxel)
                        {
                            float3 const local_coord = {.x = (float)(x - pivot.x),
                                                        .y = (float)(y - pivot.y),
                                                        .z = (float)(z - pivot.z)};
                            float3 const global_coord = vx_transform_point(&transform, local_coord);

                            int const gx = vx_round_to_int(global_coord.x);
                            int const gy = vx_round_to_int(global_coord.y);
                            int const gz = vx_round_to_int(global_coord.z);
                            int const dx = gx - scene_min.x;
                            int const dy = gy - scene_min.y;
                            int const dz = gz - scene_min.z;
                            assert(dx >= 0 && dx < grid_ext);
                            assert(dy >= 0 && dy < grid_ext);
                            assert(dz >= 0 && dz < grid_ext);

                            int const dest_idx = dx + dy * grid_ext + dz * grid_ext * grid_ext;
                            assert(dest_idx >= 0 && dest_idx < voxel_grid.count);
                            voxel_grid.ptr[dest_idx] = voxel;

                            int const macro_x = dx / VX_MACRO_CELL_EXT;
                            int const macro_y = dy / VX_MACRO_CELL_EXT;
                            int const macro_z = dz / VX_MACRO_CELL_EXT;
                            int const macro_idx =
                                vx_grid_index(macro_x, macro_y, macro_z, macro_grid_ext);
                            assert(macro_idx >= 0 && macro_idx < macro_grid.count);
                            macro_grid.ptr[macro_idx] = 1;
                        }
                    }
                }
            }
        }

        out_scene->grid_ext = grid_ext;
        out_scene->macro_grid_ext = macro_grid_ext;
        out_scene->center =
            float3(0.5f * (float)scene_ext_x, 0.5f * (float)scene_ext_y, 0.5f * (float)scene_ext_z);
    }

    out_scene->voxel_grid = voxel_grid;
    out_scene->macro_grid = macro_grid;
    for (int i = 0; i < 256; ++i)
    {
        cvox_rgba const color = scene->palette.color[i];
        out_scene->palette[i] =
            (uint)color.r | ((uint)color.g << 8u) | ((uint)color.b << 16u) | ((uint)color.a << 24u);
    }
    cvox_destroy_scene(scene);
    return true;

cleanup_grids:
    vx_buffer_free(macro_grid);
    vx_buffer_free(voxel_grid);
cleanup_scene:
    assert(scene);
    cvox_destroy_scene(scene);
    return false;
}

static void vx_scene_free(vx_scene* const scene)
{
    assert(scene);
    vx_buffer_free(scene->macro_grid);
    vx_buffer_free(scene->voxel_grid);
    scene->macro_grid = (vx_buffer(uint8_t)){0};
    scene->voxel_grid = (vx_buffer(uint8_t)){0};
}

static bool vx_gpu_buffer_upload(SDL_GPUDevice* const device, SDL_GPUBuffer* const buffer,
                                 void const* const data, uint32_t const size)
{
    SDL_GPUTransferBuffer* const transfer = SDL_CreateGPUTransferBuffer(
        device, &(SDL_GPUTransferBufferCreateInfo){.usage = SDL_GPU_TRANSFERBUFFERUSAGE_UPLOAD,
                                                   .size = size});
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
                                  void* const data, uint32_t const size, uint32_t grid_ext)
{
    SDL_GPUTransferBuffer* const transfer = SDL_CreateGPUTransferBuffer(
        device, &(SDL_GPUTransferBufferCreateInfo){.usage = SDL_GPU_TRANSFERBUFFERUSAGE_UPLOAD,
                                                   .size = size});
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

typedef struct vx_camera
{
    float3 position;
    float  yaw;
    float  pitch;
    bool   mouse_dragging;
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

static void vx_camera_basis(vx_camera const* const camera, float3* const right, float3* const up,
                            float3* const forward)
{
    float const cos_pitch = cosf(camera->pitch);
    *forward = vx_float3_normalize(
        float3(sinf(camera->yaw) * cos_pitch, sinf(camera->pitch), cosf(camera->yaw) * cos_pitch));

    float3 const world_up = float3(0.f, 1.f, 0.f);
    *right = vx_float3_normalize(vx_float3_cross(*forward, world_up));
    *up = vx_float3_cross(*right, *forward);
}

static void vx_camera_look_at(vx_camera* const camera, float3 const target)
{
    float3 const forward = vx_float3_normalize(vx_float3_sub(target, camera->position));
    camera->pitch = asinf(SDL_clamp(forward.y, -1.f, 1.f));
    camera->yaw = atan2f(forward.x, forward.z);
}

static void vx_camera_update_movement(vx_camera* const camera)
{
    float3 right = {0};
    float3 up = {0};
    float3 forward = {0};
    vx_camera_basis(camera, &right, &up, &forward);

    bool const* const keys = SDL_GetKeyboardState(0);
    float const       move_speed = 1.0f;
    float3            move = {0};
    if (keys[SDL_SCANCODE_W])
    {
        move = vx_float3_add(move, forward);
    }
    if (keys[SDL_SCANCODE_S])
    {
        move = vx_float3_sub(move, forward);
    }
    if (keys[SDL_SCANCODE_D])
    {
        move = vx_float3_add(move, right);
    }
    if (keys[SDL_SCANCODE_A])
    {
        move = vx_float3_sub(move, right);
    }
    if (keys[SDL_SCANCODE_E])
    {
        move = vx_float3_add(move, up);
    }
    if (keys[SDL_SCANCODE_Q])
    {
        move = vx_float3_sub(move, up);
    }
    if (vx_float3_norm(move) > 0.f)
    {
        camera->position =
            vx_float3_add(camera->position, vx_float3_scale(vx_float3_normalize(move), move_speed));
    }
}

typedef struct vxray
{
    // Platform
    SDL_GPUDevice* gpu_device;
    SDL_Window*    window;
    bool           window_claimed;

    // Camera
    vx_camera camera;

    // Voxel grid
    int grid_ext;

    // GPU
    SDL_GPUGraphicsPipeline* pipeline;
    SDL_GPUTexture*          voxel_texture;
    SDL_GPUTexture*          macro_texture;
    SDL_GPUBuffer*           palette_buffer;
} vxray;

static vxray vxray_instance = {0};

SDL_AppResult SDL_AppInit(void** const appstate, int const argc, char* argv[])
{
    (void)appstate;

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
            SDL_CreateGPUDevice(GPU_SHADER_FORMAT, false, GPU_DRIVER_NAME);

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

    // Graphics pipeline

    {
        SDL_GPUShaderCreateInfo const vs_info = {.code_size = DDA_VS_SIZE,
                                                 .code = DDA_VS_BYTES,
                                                 .entrypoint = GPU_SHADER_ENTRYPOINT,
                                                 .format = GPU_SHADER_FORMAT,
                                                 .stage = SDL_GPU_SHADERSTAGE_VERTEX,
                                                 .num_samplers = 0,
                                                 .num_storage_textures = 0,
                                                 .num_storage_buffers = 0,
                                                 .num_uniform_buffers = 0};
        SDL_GPUShaderCreateInfo const ps_info = {.code_size = DDA_PS_SIZE,
                                                 .code = DDA_PS_BYTES,
                                                 .entrypoint = GPU_SHADER_ENTRYPOINT,
                                                 .format = GPU_SHADER_FORMAT,
                                                 .stage = SDL_GPU_SHADERSTAGE_FRAGMENT,
                                                 .num_samplers = 0,
                                                 .num_storage_textures = 2,
                                                 .num_storage_buffers = 1,
                                                 .num_uniform_buffers = 1};
        SDL_GPUShader* const          vertex_shader =
            SDL_CreateGPUShader(vxray_instance.gpu_device, &vs_info);
        SDL_GPUShader* const fragment_shader =
            SDL_CreateGPUShader(vxray_instance.gpu_device, &ps_info);
        if (!vertex_shader)
        {
            SDL_LogError(SDL_LOG_CATEGORY_GPU, "Couldn't create vertex shader: %s", SDL_GetError());
            return SDL_APP_FAILURE;
        }
        if (!fragment_shader)
        {
            SDL_LogError(SDL_LOG_CATEGORY_GPU, "Couldn't create fragment shader: %s",
                         SDL_GetError());
            SDL_ReleaseGPUShader(vxray_instance.gpu_device, vertex_shader);
            return SDL_APP_FAILURE;
        }

        SDL_GPUGraphicsPipelineCreateInfo const pipeline_info = {
            .vertex_shader = vertex_shader,
            .fragment_shader = fragment_shader,
            .primitive_type = SDL_GPU_PRIMITIVETYPE_TRIANGLELIST,
            .rasterizer_state = (SDL_GPURasterizerState){.fill_mode = SDL_GPU_FILLMODE_FILL,
                                                         .cull_mode = SDL_GPU_CULLMODE_BACK,
                                                         .front_face = SDL_GPU_FRONTFACE_CLOCKWISE},
            .target_info = (SDL_GPUGraphicsPipelineTargetInfo){
                .num_color_targets = 1,
                .color_target_descriptions =
                    (SDL_GPUColorTargetDescription[]){
                        {.format = SDL_GetGPUSwapchainTextureFormat(vxray_instance.gpu_device,
                                                                    vxray_instance.window)}},
            }};
        SDL_GPUGraphicsPipeline* const pipeline =
            SDL_CreateGPUGraphicsPipeline(vxray_instance.gpu_device, &pipeline_info);
        if (!pipeline)
        {
            SDL_LogError(SDL_LOG_CATEGORY_GPU, "Couldn't create GPU graphics pipeline: %s",
                         SDL_GetError());
            SDL_ReleaseGPUShader(vxray_instance.gpu_device, fragment_shader);
            SDL_ReleaseGPUShader(vxray_instance.gpu_device, vertex_shader);
            return SDL_APP_FAILURE;
        }
        vxray_instance.pipeline = pipeline;

        SDL_ReleaseGPUShader(vxray_instance.gpu_device, fragment_shader);
        SDL_ReleaseGPUShader(vxray_instance.gpu_device, vertex_shader);
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
            assert(scene.grid_ext);
            vxray_instance.grid_ext = scene.grid_ext;
        }

        {
            SDL_GPUDevice* const device = vxray_instance.gpu_device;

            if (!SDL_GPUTextureSupportsFormat(vxray_instance.gpu_device,
                                              SDL_GPU_TEXTUREFORMAT_R8_UINT, SDL_GPU_TEXTURETYPE_3D,
                                              SDL_GPU_TEXTUREUSAGE_GRAPHICS_STORAGE_READ))
            {
                SDL_LogError(SDL_LOG_CATEGORY_GPU,
                             "TEXTUREFORMAT_R8_UINT not supported on this device");
                vx_scene_free(&scene);
                return SDL_APP_FAILURE;
            }

            {
                uint32_t const        voxel_buffer_size = (uint32_t)scene.voxel_grid.count;
                SDL_GPUTexture* const voxel_texture = SDL_CreateGPUTexture(
                    device,
                    &(SDL_GPUTextureCreateInfo){.type = SDL_GPU_TEXTURETYPE_3D,
                                                .format = SDL_GPU_TEXTUREFORMAT_R8_UINT,
                                                .usage = SDL_GPU_TEXTUREUSAGE_GRAPHICS_STORAGE_READ,
                                                .width = scene.grid_ext,
                                                .height = scene.grid_ext,
                                                .layer_count_or_depth = scene.grid_ext,
                                                .num_levels = 1,
                                                .sample_count = SDL_GPU_SAMPLECOUNT_1});
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
                uint32_t const        macro_grid_size = (uint32_t)scene.macro_grid.count;
                SDL_GPUTexture* const macro_texture = SDL_CreateGPUTexture(
                    device,
                    &(SDL_GPUTextureCreateInfo){.type = SDL_GPU_TEXTURETYPE_3D,
                                                .format = SDL_GPU_TEXTUREFORMAT_R8_UINT,
                                                .usage = SDL_GPU_TEXTUREUSAGE_GRAPHICS_STORAGE_READ,
                                                .width = scene.macro_grid_ext,
                                                .height = scene.macro_grid_ext,
                                                .layer_count_or_depth = scene.macro_grid_ext,
                                                .num_levels = 1,
                                                .sample_count = SDL_GPU_SAMPLECOUNT_1});
                if (!macro_texture)
                {
                    SDL_LogError(SDL_LOG_CATEGORY_GPU, "Failed to create macro grid texture: %s",
                                 SDL_GetError());
                    vx_scene_free(&scene);
                    return SDL_APP_FAILURE;
                }
                if (!vx_gpu_texture_upload(device, macro_texture, scene.macro_grid.ptr,
                                           macro_grid_size, (uint32_t)scene.macro_grid_ext))
                {
                    vx_scene_free(&scene);
                    return SDL_APP_FAILURE;
                }
                vxray_instance.macro_texture = macro_texture;
            }
            {
                uint32_t const palette_size = 4 * 256;
                assert(palette_size == sizeof(scene.palette));
                SDL_GPUBuffer* const palette_buffer = SDL_CreateGPUBuffer(
                    device,
                    &(SDL_GPUBufferCreateInfo){.usage = SDL_GPU_BUFFERUSAGE_GRAPHICS_STORAGE_READ,
                                               .size = palette_size});
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
                vx_float3_add(scene.center, float3(0.f, view_radius * 0.3f, -view_radius * 2.8f));
            vx_camera_look_at(&vxray_instance.camera, scene.center);
        }
    }

    return SDL_APP_CONTINUE;
}

SDL_AppResult SDL_AppEvent(void* const appstate, SDL_Event* const event)
{
    (void)appstate;

    if (event->type == SDL_EVENT_QUIT)
    {
        return SDL_APP_SUCCESS;
    }
    if (event->type == SDL_EVENT_KEY_DOWN && !event->key.repeat &&
        event->key.scancode == SDL_SCANCODE_F2)
    {
        vx_camera_print_code(&vxray_instance.camera);
    }
    if (event->type == SDL_EVENT_MOUSE_BUTTON_DOWN && event->button.button == SDL_BUTTON_LEFT)
    {
        vxray_instance.camera.mouse_dragging = true;
        if (!SDL_SetWindowRelativeMouseMode(vxray_instance.window, true))
        {
            SDL_LogWarn(SDL_LOG_CATEGORY_INPUT, "Couldn't enable relative mouse mode: %s",
                        SDL_GetError());
        }
    }
    if (event->type == SDL_EVENT_MOUSE_BUTTON_UP && event->button.button == SDL_BUTTON_LEFT)
    {
        vxray_instance.camera.mouse_dragging = false;
        if (!SDL_SetWindowRelativeMouseMode(vxray_instance.window, false))
        {
            SDL_LogWarn(SDL_LOG_CATEGORY_INPUT, "Couldn't disable relative mouse mode: %s",
                        SDL_GetError());
        }
    }
    if (event->type == SDL_EVENT_MOUSE_MOTION && vxray_instance.camera.mouse_dragging)
    {
        vx_camera* const camera = &vxray_instance.camera;
        float const      mouse_sensitivity = 0.003f;
        float const      pitch_limit = 1.55334306f;
        camera->yaw += (float)event->motion.xrel * mouse_sensitivity;
        camera->pitch = SDL_clamp(camera->pitch - (float)event->motion.yrel * mouse_sensitivity,
                                  -pitch_limit, pitch_limit);
    }

    return SDL_APP_CONTINUE;
}

SDL_AppResult SDL_AppIterate(void* const appstate)
{
    (void)appstate;

    SDL_GPUDevice* const gpu_device = vxray_instance.gpu_device;
    SDL_Window* const    window = vxray_instance.window;
    vx_camera* const     camera = &vxray_instance.camera;
    vx_camera_update_movement(camera);

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

    SDL_GPUColorTargetInfo const color_target_info = {.texture = swapchain_texture,
                                                      .clear_color =
                                                          (SDL_FColor){0.f, 0.f, 0.f, 0.f},
                                                      .load_op = SDL_GPU_LOADOP_CLEAR,
                                                      .store_op = SDL_GPU_STOREOP_STORE};
    SDL_GPURenderPass* render_pass = SDL_BeginGPURenderPass(cmd_buffer, &color_target_info, 1, 0);
    assert(render_pass);

    SDL_BindGPUGraphicsPipeline(render_pass, vxray_instance.pipeline);
    SDL_GPUTexture* textures[] = {
        vxray_instance.voxel_texture,
        vxray_instance.macro_texture,
    };
    SDL_BindGPUFragmentStorageTextures(render_pass, 0, textures, SDL_arraysize(textures));
    SDL_GPUBuffer* buffers[] = {
        vxray_instance.palette_buffer,
    };
    SDL_BindGPUFragmentStorageBuffers(render_pass, 0, buffers, SDL_arraysize(buffers));

    float3 right = {0};
    float3 up = {0};
    float3 forward = {0};
    vx_camera_basis(camera, &right, &up, &forward);
    dda_uniforms const uniforms = {.camera_pos = vx_float4_from_float3(camera->position, 0.f),
                                   .camera_right = vx_float4_from_float3(right, 0.f),
                                   .camera_up = vx_float4_from_float3(up, 0.f),
                                   .camera_forward = vx_float4_from_float3(forward, 0.f),
                                   .viewport = {(float)width, (float)height, 60.f, 0.f},
                                   .grid_ext = vxray_instance.grid_ext};
    SDL_PushGPUFragmentUniformData(cmd_buffer, 0, &uniforms, sizeof(dda_uniforms));
    SDL_DrawGPUPrimitives(render_pass, 6, 1, 0, 0);
    SDL_EndGPURenderPass(render_pass);
    if (!SDL_SubmitGPUCommandBuffer(cmd_buffer))
    {
        SDL_LogError(SDL_LOG_CATEGORY_GPU, "Couldn't submit GPU command buffer: %s",
                     SDL_GetError());
        return SDL_APP_FAILURE;
    }

    return SDL_APP_CONTINUE;
}

void SDL_AppQuit(void* const appstate, SDL_AppResult const result)
{
    (void)appstate;
    (void)result;

    if (vxray_instance.palette_buffer)
    {
        SDL_ReleaseGPUBuffer(vxray_instance.gpu_device, vxray_instance.palette_buffer);
        vxray_instance.palette_buffer = 0;
    }

    if (vxray_instance.macro_texture)
    {
        SDL_ReleaseGPUTexture(vxray_instance.gpu_device, vxray_instance.macro_texture);
        vxray_instance.macro_texture = 0;
    }

    if (vxray_instance.voxel_texture)
    {
        SDL_ReleaseGPUTexture(vxray_instance.gpu_device, vxray_instance.voxel_texture);
        vxray_instance.voxel_texture = 0;
    }

    if (vxray_instance.pipeline)
    {
        SDL_ReleaseGPUGraphicsPipeline(vxray_instance.gpu_device, vxray_instance.pipeline);
        vxray_instance.pipeline = 0;
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
