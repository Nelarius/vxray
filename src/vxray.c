#include "cvox.h"
#include "dda.h"

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

typedef struct vxray
{
    SDL_GPUDevice*           gpu_device;
    SDL_Window*              window;
    SDL_GPUGraphicsPipeline* pipeline;

    bool window_claimed;
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
        char const* vox_file = argv[1];

        SDL_PathInfo info;
        if (!SDL_GetPathInfo(vox_file, &info))
        {
            SDL_LogError(SDL_LOG_CATEGORY_ERROR, "%s doesn't exist", vox_file);
            return SDL_APP_FAILURE;
        }

        size_t   num_bytes;
        uint8_t* buffer = SDL_LoadFile(vox_file, &num_bytes);
        assert(buffer);

        cvox_scene const* const scene = cvox_read_scene(buffer, num_bytes);
        (void)scene;

        cvox_destroy_scene(scene);
        SDL_free(buffer);
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
        SDL_GPUShaderCreateInfo const vs_info = {.code_size = FULLSCREEN_VS_SIZE,
                                                 .code = FULLSCREEN_VS_BYTES,
                                                 .entrypoint = GPU_SHADER_ENTRYPOINT,
                                                 .format = GPU_SHADER_FORMAT,
                                                 .stage = SDL_GPU_SHADERSTAGE_VERTEX,
                                                 .num_samplers = 0,
                                                 .num_storage_textures = 0,
                                                 .num_storage_buffers = 0,
                                                 .num_uniform_buffers = 0};
        SDL_GPUShaderCreateInfo const ps_info = {.code_size = FULLSCREEN_PS_SIZE,
                                                 .code = FULLSCREEN_PS_BYTES,
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
