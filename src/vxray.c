#include "cvox.h"
#include "hlsl_shim.h"

#define SDL_MAIN_USE_CALLBACKS 1
#include <SDL3/SDL_error.h>
#include <SDL3/SDL_events.h>
#include <SDL3/SDL_filesystem.h>
#include <SDL3/SDL_gpu.h>
#include <SDL3/SDL_init.h>
#include <SDL3/SDL_iostream.h>
#include <SDL3/SDL_log.h>
#include <SDL3/SDL_main.h>
#include <SDL3/SDL_pixels.h>
#include <SDL3/SDL_platform_defines.h>
#include <SDL3/SDL_stdinc.h>
#include <SDL3/SDL_video.h>

#include <assert.h>
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

static uint32_t next_power_of_2(uint32_t x)
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
    float3 const result = {.x = t->m30 + (t->m00 * p.x) + (t->m10 * p.y) + (t->m20 * p.z),
                           .y = t->m31 + (t->m01 * p.x) + (t->m11 * p.y) + (t->m21 * p.z),
                           .z = t->m32 + (t->m02 * p.x) + (t->m12 * p.y) + (t->m22 * p.z)};
    return result;
}

static int vx_round_to_int(float const val)
{
    return (int)(val >= 0.f ? (val + 0.5f) : (val - 0.5f));
}

static int vx_grid_index(int const x, int const y, int const z, int const grid_ext)
{
    return x + (y * grid_ext) + (z * grid_ext * grid_ext);
}

typedef struct vxray
{
    // Platform
    SDL_GPUDevice* gpu_device;
    SDL_Window*    window;
    bool           window_claimed;

    // GPU
    SDL_GPUGraphicsPipeline* pipeline;
} vxray;

static vxray vxray_instance = {0};

SDL_AppResult SDL_AppInit(void** const appstate, int const argc, char* argv[])
{
    (void)appstate;

    if (argc < 2)
    {
        fprintf(stderr, "Usage: %s <file.vox>\n", argv[0]);
        return SDL_APP_FAILURE;
    }

    {
        cvox_scene const* scene = 0;
        {
            char const* vox_file = argv[1];

            SDL_PathInfo info;
            if (!SDL_GetPathInfo(vox_file, &info))
            {
                SDL_LogError(SDL_LOG_CATEGORY_APPLICATION, "%s doesn't exist", vox_file);
                return SDL_APP_FAILURE;
            }

            size_t   num_bytes;
            uint8_t* buffer = SDL_LoadFile(vox_file, &num_bytes);
            if (!buffer || !num_bytes)
            {
                SDL_LogError(SDL_LOG_CATEGORY_APPLICATION, "Failed to load %s", vox_file);
                return SDL_APP_FAILURE;
            }

            scene = cvox_read_scene(buffer, num_bytes);

            SDL_free(buffer);
        }
        if (scene == 0)
        {
            SDL_LogError(SDL_LOG_CATEGORY_APPLICATION, "Failed to load scene from %s", argv[1]);
            return SDL_APP_FAILURE;
        }
        if (scene->num_instances == 0 || scene->num_models == 0)
        {
            SDL_LogError(SDL_LOG_CATEGORY_APPLICATION, "Scene has no instances or models");
            cvox_destroy_scene(scene);
            return SDL_APP_FAILURE;
        }

        int3 scene_min = {.x = INT32_MAX, .y = INT32_MAX, .z = INT32_MAX};
        int3 scene_max = {.x = INT32_MIN, .y = INT32_MIN, .z = INT32_MIN};
        bool has_valid_bounds = false;

        for (int i = 0; i < scene->num_instances; ++i)
        {
            cvox_instance const* const instance = &scene->instances[i];
            uint32_t const             model_index = cvox_sample_instance_model(instance, 0);
            if (model_index >= scene->num_models || scene->models[model_index] == 0)
            {
                continue;
            }

            cvox_model const* const model = scene->models[model_index];
            cvox_transform const    transform =
                cvox_sample_instance_transform_global(instance, 0, scene);

            float const size_x_f = (float)model->size_x;
            float const size_y_f = (float)model->size_y;
            float const size_z_f = (float)model->size_z;

            float const max_x = size_x_f > 0.0f ? (size_x_f - 1.0f) : 0.0f;
            float const max_y = size_y_f > 0.0f ? (size_y_f - 1.0f) : 0.0f;
            float const max_z = size_z_f > 0.0f ? (size_z_f - 1.0f) : 0.0f;

            float3 const corners[8] = {{0.f, 0.f, 0.f},     {max_x, 0.f, 0.f},
                                       {0.f, max_y, 0.f},   {0.f, 0.f, max_z},
                                       {max_x, max_y, 0.f}, {max_x, 0.f, max_z},
                                       {0.f, max_y, max_z}, {max_x, max_y, max_z}};

            for (int c = 0; c < 8; ++c)
            {
                float3 const tc = vx_transform_point(&transform, corners[c]);
                int const    rx = vx_round_to_int(tc.x);
                int const    ry = vx_round_to_int(tc.y);
                int const    rz = vx_round_to_int(tc.z);

                if (rx < scene_min.x)
                {
                    scene_min.x = rx;
                }
                if (rx > scene_max.x)
                {
                    scene_max.x = rx;
                }
                if (ry < scene_min.y)
                {
                    scene_min.y = ry;
                }
                if (ry > scene_max.y)
                {
                    scene_max.y = ry;
                }
                if (rz < scene_min.z)
                {
                    scene_min.z = rz;
                }
                if (rz > scene_max.z)
                {
                    scene_max.z = rz;
                }
            }

            has_valid_bounds = true;
        }

        if (!has_valid_bounds)
        {
            SDL_LogError(SDL_LOG_CATEGORY_APPLICATION,
                         "Scene has no valid instances with non-empty models");
            cvox_destroy_scene(scene);
            return SDL_APP_FAILURE;
        }

        int const extent_x = scene_max.x - scene_min.x + 1;
        int const extent_y = scene_max.y - scene_min.y + 1;
        int const extent_z = scene_max.z - scene_min.z + 1;

        if (extent_x <= 0 || extent_y <= 0 || extent_z <= 0)
        {
            SDL_LogError(SDL_LOG_CATEGORY_APPLICATION, "Invalid scene extents: %d x %d x %d",
                         extent_x, extent_y, extent_z);
            cvox_destroy_scene(scene);
            return SDL_APP_FAILURE;
        }

        int const largest_extent = (extent_x > extent_y)
                                       ? (extent_x > extent_z ? extent_x : extent_z)
                                       : (extent_y > extent_z ? extent_y : extent_z);

        uint32_t const grid_ext = next_power_of_2((uint32_t)largest_extent);
        if (grid_ext > 1024)
        {
            SDL_LogError(SDL_LOG_CATEGORY_APPLICATION,
                         "Requested voxel grid size is too large: %u^3", grid_ext);
            cvox_destroy_scene(scene);
            return SDL_APP_FAILURE;
        }
        int const total_voxels = (int)grid_ext * grid_ext * grid_ext;

        vx_buffer(uint8_t) const voxel_grid = vx_buffer_calloc(uint8_t, total_voxels);
        if (voxel_grid.ptr == 0)
        {
            SDL_LogError(SDL_LOG_CATEGORY_ERROR,
                         "Failed to allocate memory for voxel grid of size %u^3", grid_ext);
            cvox_destroy_scene(scene);
            return SDL_APP_FAILURE;
        }

        for (uint32_t i = 0; i < scene->num_instances; ++i)
        {
            cvox_instance const* const instance = &scene->instances[i];
            uint32_t const             model_index = cvox_sample_instance_model(instance, 0);
            if (model_index >= scene->num_models || scene->models[model_index] == 0)
            {
                continue;
            }

            cvox_model const* const model = scene->models[model_index];
            cvox_transform const    transform =
                cvox_sample_instance_transform_global(instance, 0, scene);

            uint32_t const sx = model->size_x;
            uint32_t const sy = model->size_y;
            uint32_t const sz = model->size_z;

            for (uint32_t z = 0; z < sz; ++z)
            {
                for (uint32_t y = 0; y < sy; ++y)
                {
                    for (uint32_t x = 0; x < sx; ++x)
                    {
                        uint32_t const src_index = x + (y * sx) + (z * sx * sy);
                        uint8_t const  voxel_val = model->voxel_data[src_index];

                        if (voxel_val != 0)
                        {
                            float3 const local_p = {(float)x, (float)y, (float)z};
                            float3 const trans_p = vx_transform_point(&transform, local_p);
                            int const    tx = vx_round_to_int(trans_p.x);
                            int const    ty = vx_round_to_int(trans_p.y);
                            int const    tz = vx_round_to_int(trans_p.z);

                            int const dest_x = tx - scene_min.x;
                            int const dest_y = ty - scene_min.y;
                            int const dest_z = tz - scene_min.z;

                            if (dest_x >= 0 && dest_x < (int)grid_ext && dest_y >= 0 &&
                                dest_y < (int)grid_ext && dest_z >= 0 && dest_z < (int)grid_ext)
                            {
                                int const dest_index =
                                    vx_grid_index(dest_x, dest_y, dest_z, (int)grid_ext);
                                voxel_grid.ptr[dest_index] = voxel_val;
                            }
                        }
                    }
                }
            }
        }

        (void)voxel_grid;

        vx_buffer_free(voxel_grid);
        cvox_destroy_scene(scene);
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
                                                 .num_storage_textures = 0,
                                                 .num_storage_buffers = 0,
                                                 .num_uniform_buffers = 0};
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

    return SDL_APP_CONTINUE;
}

SDL_AppResult SDL_AppEvent(void* const appstate, SDL_Event* const event)
{
    (void)appstate;

    if (event->type == SDL_EVENT_QUIT)
    {
        return SDL_APP_SUCCESS;
    }

    return SDL_APP_CONTINUE;
}

SDL_AppResult SDL_AppIterate(void* const appstate)
{
    (void)appstate;

    SDL_GPUDevice* const gpu_device = vxray_instance.gpu_device;
    SDL_Window* const    window = vxray_instance.window;

    SDL_GPUCommandBuffer* const cmd_buffer = SDL_AcquireGPUCommandBuffer(gpu_device);
    if (!cmd_buffer)
    {
        SDL_LogError(SDL_LOG_CATEGORY_GPU, "Couldn't acquire GPU command buffer: %s\n",
                     SDL_GetError());
        return SDL_APP_FAILURE;
    }

    SDL_GPUTexture* swapchain_texture = 0;
    if (!SDL_WaitAndAcquireGPUSwapchainTexture(cmd_buffer, window, &swapchain_texture, 0, 0))
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

    if (vxray_instance.pipeline)
    {
        SDL_ReleaseGPUGraphicsPipeline(vxray_instance.gpu_device, vxray_instance.pipeline);
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
