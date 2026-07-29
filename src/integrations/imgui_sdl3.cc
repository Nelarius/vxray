#include "imgui_sdl3.h"

#include "imgui.h"
#include "imgui_impl_sdl3.h"
#include "imgui_impl_sdlgpu3.h"

#include <SDL3/SDL_error.h>
#include <SDL3/SDL_gpu.h>
#include <SDL3/SDL_video.h>

static SDL_GPUDevice* gpu_device = nullptr;
static bool           platform_initialized = false;
static bool           renderer_initialized = false;

extern "C" bool imgui_sdl3_init(SDL_GPUDevice* const device, SDL_Window* const window)
{
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGui::StyleColorsDark();

    gpu_device = device;
    if (!ImGui_ImplSDL3_InitForSDLGPU(window))
    {
        imgui_sdl3_shutdown();
        return SDL_SetError("Failed to initialize the ImGui SDL3 platform backend");
    }
    platform_initialized = true;

    ImGui_ImplSDLGPU3_InitInfo init_info = {};
    init_info.Device = device;
    init_info.ColorTargetFormat = SDL_GetGPUSwapchainTextureFormat(device, window);
    init_info.MSAASamples = SDL_GPU_SAMPLECOUNT_1;
    init_info.SwapchainComposition = SDL_GPU_SWAPCHAINCOMPOSITION_SDR_LINEAR;
    init_info.PresentMode = SDL_GPU_PRESENTMODE_VSYNC;
    if (!ImGui_ImplSDLGPU3_Init(&init_info))
    {
        imgui_sdl3_shutdown();
        return SDL_SetError("Failed to initialize the ImGui SDL_GPU renderer backend");
    }
    renderer_initialized = true;

    return true;
}

extern "C" bool imgui_sdl3_process_event(SDL_Event const* const event)
{
    return ImGui_ImplSDL3_ProcessEvent(event);
}

extern "C" void imgui_sdl3_new_frame()
{
    ImGui_ImplSDLGPU3_NewFrame();
    ImGui_ImplSDL3_NewFrame();
    ImGui::NewFrame();
}

extern "C" bool imgui_sdl3_wants_capture_keyboard() { return ImGui::GetIO().WantCaptureKeyboard; }

extern "C" bool imgui_sdl3_wants_capture_mouse() { return ImGui::GetIO().WantCaptureMouse; }

extern "C" void imgui_sdl3_prepare_draw_data(SDL_GPUCommandBuffer* const command_buffer)
{
    ImGui_ImplSDLGPU3_PrepareDrawData(ImGui::GetDrawData(), command_buffer);
}

extern "C" void imgui_sdl3_render_draw_data(SDL_GPUCommandBuffer* const command_buffer,
                                            SDL_GPURenderPass* const    render_pass)
{
    ImGui_ImplSDLGPU3_RenderDrawData(ImGui::GetDrawData(), command_buffer, render_pass);
}

extern "C" void imgui_sdl3_shutdown()
{
    if (gpu_device)
    {
        SDL_WaitForGPUIdle(gpu_device);
    }
    if (renderer_initialized)
    {
        ImGui_ImplSDLGPU3_Shutdown();
        renderer_initialized = false;
    }
    if (platform_initialized)
    {
        ImGui_ImplSDL3_Shutdown();
        platform_initialized = false;
    }
    if (ImGui::GetCurrentContext())
    {
        ImGui::DestroyContext();
    }
    gpu_device = nullptr;
}
