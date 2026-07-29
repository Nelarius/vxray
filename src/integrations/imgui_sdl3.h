#ifndef VXRAY_IMGUI_SDL3_H
#define VXRAY_IMGUI_SDL3_H

#include <SDL3/SDL_events.h>
#include <SDL3/SDL_gpu.h>
#include <SDL3/SDL_video.h>

#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

bool imgui_sdl3_init(SDL_GPUDevice* device, SDL_Window* window);
bool imgui_sdl3_process_event(SDL_Event const* event);
void imgui_sdl3_new_frame(void);
bool imgui_sdl3_wants_capture_keyboard(void);
bool imgui_sdl3_wants_capture_mouse(void);
void imgui_sdl3_prepare_draw_data(SDL_GPUCommandBuffer* command_buffer);
void imgui_sdl3_render_draw_data(SDL_GPUCommandBuffer* command_buffer,
                                 SDL_GPURenderPass*    render_pass);
void imgui_sdl3_shutdown(void);

#ifdef __cplusplus
}
#endif

#endif
