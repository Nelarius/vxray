#include "brick_quad.h"
#include "constants.h"
#include "cvox.h"
#include "display.h"
#include "gbuffer.h"
#include "hlsl_shim.h"
#include "rtao.h"

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
static_assert(sizeof(brick_quad_uniforms) == 96, "brick-quad uniform layout must match HLSL");
static_assert(sizeof(display_uniforms) == 96, "display uniform layout must match HLSL");
static_assert(sizeof(gbuffer_uniforms) == 160, "G-buffer uniform layout must match HLSL");
static_assert(sizeof(rtao_uniforms) == 128, "RTAO uniform layout must match HLSL");
static_assert(1024 / VX_BRICK_EXT <= 256, "brick-grid coordinates must fit in packed face records");

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
vx_buffer_decl(uint32_t);

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

static int vx_grid_index(int const x, int const y, int const z, int const grid_ext)
{
    return x + (y * grid_ext) + (z * grid_ext * grid_ext);
}

typedef struct vx_scene
{
    vx_buffer(uint8_t) voxel_grid;
    vx_buffer(uint8_t) brick_grid;
    uint  palette[256];
    int   grid_ext;
    int   brick_grid_ext;
    vec3s center;
} vx_scene;

// Loads a MagicaVoxel scene into dense voxel and brick grids. Free both grids with
// `vx_buffer_free`.
static bool vx_load_scene(char const* const vox_path, vx_scene* const out_scene)
{
    assert(vox_path);
    assert(out_scene);

    vx_buffer(uint8_t) voxel_grid = {0};
    vx_buffer(uint8_t) brick_grid = {0};
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
        if (largest_extent > 1024) // NOTE: 1024^ 3 is (1 << 30)
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
                            vec3s const local_coord = {.x = (float)(x - pivot.x),
                                                       .y = (float)(y - pivot.y),
                                                       .z = (float)(z - pivot.z)};
                            vec3s const global_coord = glms_mat4_mulv3(transform, local_coord, 1.f);

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

                            int const brick_x = dx / VX_BRICK_EXT;
                            int const brick_y = dy / VX_BRICK_EXT;
                            int const brick_z = dz / VX_BRICK_EXT;
                            int const brick_idx =
                                vx_grid_index(brick_x, brick_y, brick_z, brick_grid_ext);
                            assert(brick_idx >= 0 && brick_idx < brick_grid.count);
                            brick_grid.ptr[brick_idx] = 1;
                        }
                    }
                }
            }
        }

        out_scene->grid_ext = grid_ext;
        out_scene->brick_grid_ext = brick_grid_ext;
        out_scene->center = (vec3s){0.5f * (float)scene_ext_x, 0.5f * (float)scene_ext_y,
                                    0.5f * (float)scene_ext_z};
    }

    out_scene->voxel_grid = voxel_grid;
    out_scene->brick_grid = brick_grid;
    for (int i = 0; i < 256; ++i)
    {
        cvox_rgba const color = scene->palette.color[i];
        out_scene->palette[i] =
            (uint)color.r | ((uint)color.g << 8u) | ((uint)color.b << 16u) | ((uint)color.a << 24u);
    }
    cvox_destroy_scene(scene);
    return true;

cleanup_grids:
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
    vx_buffer_free(scene->brick_grid);
    vx_buffer_free(scene->voxel_grid);
    scene->brick_grid = (vx_buffer(uint8_t)){0};
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
    float     ao_sp;
    float     ao_smin;
    uint32_t  frame_index;

    // Voxel grid
    int      grid_ext;
    int      brick_grid_ext;
    uint32_t face_capacity;

    // GPU
    SDL_GPUGraphicsPipeline* gbuffer_pipeline;
    SDL_GPUComputePipeline*  rtao_pipeline;
    SDL_GPUGraphicsPipeline* display_pipeline;
    SDL_GPUGraphicsPipeline* brick_quad_pipeline;
    SDL_GPUComputePipeline*  brick_quad_compute_pipeline;
    SDL_GPUTexture*          voxel_texture;
    SDL_GPUTexture*          brick_texture;
    SDL_GPUTexture*          entry_depth_texture;
    SDL_GPUTexture*          entry_brick_texture;
    SDL_GPUTexture*          gbuffer_albedo_texture;
    SDL_GPUTexture*          gbuffer_normal_texture;
    SDL_GPUTexture*          gbuffer_depth_texture;
    uint32_t                 render_width;
    uint32_t                 render_height;
    SDL_GPUSampler*          display_sampler;
    SDL_GPUBuffer*           visible_faces_buffer;
    SDL_GPUBuffer*           indirect_draw_buffer;
    SDL_GPUTransferBuffer*   indirect_reset_transfer_buffer;
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

    if (vxray_instance.entry_depth_texture && vxray_instance.entry_brick_texture &&
        vxray_instance.gbuffer_albedo_texture && vxray_instance.gbuffer_normal_texture &&
        vxray_instance.gbuffer_depth_texture && vxray_instance.render_width == width &&
        vxray_instance.render_height == height)
    {
        return true;
    }

    SDL_GPUTexture* const entry_depth_texture = SDL_CreateGPUTexture(
        vxray_instance.gpu_device,
        &(SDL_GPUTextureCreateInfo){.type = SDL_GPU_TEXTURETYPE_2D,
                                    .format = SDL_GPU_TEXTUREFORMAT_D32_FLOAT,
                                    .usage = SDL_GPU_TEXTUREUSAGE_DEPTH_STENCIL_TARGET,
                                    .width = width,
                                    .height = height,
                                    .layer_count_or_depth = 1,
                                    .num_levels = 1,
                                    .sample_count = SDL_GPU_SAMPLECOUNT_1});
    if (!entry_depth_texture)
    {
        SDL_LogError(SDL_LOG_CATEGORY_GPU, "Failed to create entry depth texture: %s",
                     SDL_GetError());
        return false;
    }

    SDL_GPUTexture* const entry_brick_texture = SDL_CreateGPUTexture(
        vxray_instance.gpu_device,
        &(SDL_GPUTextureCreateInfo){.type = SDL_GPU_TEXTURETYPE_2D,
                                    .format = SDL_GPU_TEXTUREFORMAT_R32_UINT,
                                    .usage = SDL_GPU_TEXTUREUSAGE_COLOR_TARGET |
                                             SDL_GPU_TEXTUREUSAGE_GRAPHICS_STORAGE_READ,
                                    .width = width,
                                    .height = height,
                                    .layer_count_or_depth = 1,
                                    .num_levels = 1,
                                    .sample_count = SDL_GPU_SAMPLECOUNT_1});
    if (!entry_brick_texture)
    {
        SDL_LogError(SDL_LOG_CATEGORY_GPU, "Failed to create entry brick texture: %s",
                     SDL_GetError());
        SDL_ReleaseGPUTexture(vxray_instance.gpu_device, entry_depth_texture);
        return false;
    }

    SDL_GPUTexture* const gbuffer_albedo_texture = SDL_CreateGPUTexture(
        vxray_instance.gpu_device,
        &(SDL_GPUTextureCreateInfo){.type = SDL_GPU_TEXTURETYPE_2D,
                                    .format = SDL_GPU_TEXTUREFORMAT_R32_UINT,
                                    .usage = SDL_GPU_TEXTUREUSAGE_COLOR_TARGET |
                                             SDL_GPU_TEXTUREUSAGE_GRAPHICS_STORAGE_READ,
                                    .width = width,
                                    .height = height,
                                    .layer_count_or_depth = 1,
                                    .num_levels = 1,
                                    .sample_count = SDL_GPU_SAMPLECOUNT_1});
    if (!gbuffer_albedo_texture)
    {
        SDL_LogError(SDL_LOG_CATEGORY_GPU, "Failed to create G-buffer albedo texture: %s",
                     SDL_GetError());
        SDL_ReleaseGPUTexture(vxray_instance.gpu_device, entry_brick_texture);
        SDL_ReleaseGPUTexture(vxray_instance.gpu_device, entry_depth_texture);
        return false;
    }

    SDL_GPUTexture* const gbuffer_normal_texture = SDL_CreateGPUTexture(
        vxray_instance.gpu_device,
        &(SDL_GPUTextureCreateInfo){.type = SDL_GPU_TEXTURETYPE_2D,
                                    .format = SDL_GPU_TEXTUREFORMAT_R8_UINT,
                                    .usage = SDL_GPU_TEXTUREUSAGE_COLOR_TARGET |
                                             SDL_GPU_TEXTUREUSAGE_GRAPHICS_STORAGE_READ |
                                             SDL_GPU_TEXTUREUSAGE_COMPUTE_STORAGE_READ,
                                    .width = width,
                                    .height = height,
                                    .layer_count_or_depth = 1,
                                    .num_levels = 1,
                                    .sample_count = SDL_GPU_SAMPLECOUNT_1});
    if (!gbuffer_normal_texture)
    {
        SDL_LogError(SDL_LOG_CATEGORY_GPU, "Failed to create G-buffer normal texture: %s",
                     SDL_GetError());
        SDL_ReleaseGPUTexture(vxray_instance.gpu_device, gbuffer_albedo_texture);
        SDL_ReleaseGPUTexture(vxray_instance.gpu_device, entry_brick_texture);
        SDL_ReleaseGPUTexture(vxray_instance.gpu_device, entry_depth_texture);
        return false;
    }

    SDL_GPUTexture* const gbuffer_depth_texture = SDL_CreateGPUTexture(
        vxray_instance.gpu_device,
        &(SDL_GPUTextureCreateInfo){.type = SDL_GPU_TEXTURETYPE_2D,
                                    .format = SDL_GPU_TEXTUREFORMAT_D32_FLOAT,
                                    .usage = SDL_GPU_TEXTUREUSAGE_DEPTH_STENCIL_TARGET |
                                             SDL_GPU_TEXTUREUSAGE_SAMPLER,
                                    .width = width,
                                    .height = height,
                                    .layer_count_or_depth = 1,
                                    .num_levels = 1,
                                    .sample_count = SDL_GPU_SAMPLECOUNT_1});
    if (!gbuffer_depth_texture)
    {
        SDL_LogError(SDL_LOG_CATEGORY_GPU, "Failed to create G-buffer depth texture: %s",
                     SDL_GetError());
        SDL_ReleaseGPUTexture(vxray_instance.gpu_device, gbuffer_normal_texture);
        SDL_ReleaseGPUTexture(vxray_instance.gpu_device, gbuffer_albedo_texture);
        SDL_ReleaseGPUTexture(vxray_instance.gpu_device, entry_brick_texture);
        SDL_ReleaseGPUTexture(vxray_instance.gpu_device, entry_depth_texture);
        return false;
    }

    if (vxray_instance.entry_depth_texture)
    {
        SDL_ReleaseGPUTexture(vxray_instance.gpu_device, vxray_instance.entry_depth_texture);
    }
    if (vxray_instance.entry_brick_texture)
    {
        SDL_ReleaseGPUTexture(vxray_instance.gpu_device, vxray_instance.entry_brick_texture);
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
    vxray_instance.entry_depth_texture = entry_depth_texture;
    vxray_instance.entry_brick_texture = entry_brick_texture;
    vxray_instance.gbuffer_albedo_texture = gbuffer_albedo_texture;
    vxray_instance.gbuffer_normal_texture = gbuffer_normal_texture;
    vxray_instance.gbuffer_depth_texture = gbuffer_depth_texture;
    vxray_instance.render_width = width;
    vxray_instance.render_height = height;
    return true;
}

SDL_AppResult SDL_AppInit(void** const appstate, int const argc, char* argv[])
{
    (void)appstate;

    vxray_instance.ao_sp = 10.f;
    vxray_instance.ao_smin = 0.07f;

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
            SDL_GPU_TEXTUREUSAGE_COLOR_TARGET | SDL_GPU_TEXTUREUSAGE_GRAPHICS_STORAGE_READ |
                SDL_GPU_TEXTUREUSAGE_COMPUTE_STORAGE_READ))
    {
        SDL_LogError(SDL_LOG_CATEGORY_GPU, "TEXTUREFORMAT_R8_UINT not supported on this device");
        return SDL_APP_FAILURE;
    }

    // Brick-face generation pipeline

    vxray_instance.brick_quad_compute_pipeline = SDL_CreateGPUComputePipeline(
        vxray_instance.gpu_device,
        &(SDL_GPUComputePipelineCreateInfo){.code_size = BRICK_QUAD_CS_SIZE,
                                            .code = BRICK_QUAD_CS_BYTES,
                                            .entrypoint = GPU_SHADER_ENTRYPOINT,
                                            .format = GPU_SHADER_FORMAT,
                                            .num_readonly_storage_textures = 1,
                                            .num_readwrite_storage_buffers = 2,
                                            .num_uniform_buffers = 1,
                                            .threadcount_x = 64,
                                            .threadcount_y = 1,
                                            .threadcount_z = 1});
    if (!vxray_instance.brick_quad_compute_pipeline)
    {
        SDL_LogError(SDL_LOG_CATEGORY_GPU, "Couldn't create brick-quad compute pipeline: %s",
                     SDL_GetError());
        return SDL_APP_FAILURE;
    }

    // Brick-entry rasterization pipeline

    {
        SDL_GPUShaderCreateInfo const vs_info = {.code_size = BRICK_QUAD_VS_SIZE,
                                                 .code = BRICK_QUAD_VS_BYTES,
                                                 .entrypoint = GPU_SHADER_ENTRYPOINT,
                                                 .format = GPU_SHADER_FORMAT,
                                                 .stage = SDL_GPU_SHADERSTAGE_VERTEX,
                                                 .num_storage_buffers = 1,
                                                 .num_uniform_buffers = 1};
        SDL_GPUShaderCreateInfo const ps_info = {.code_size = BRICK_QUAD_PS_SIZE,
                                                 .code = BRICK_QUAD_PS_BYTES,
                                                 .entrypoint = GPU_SHADER_ENTRYPOINT,
                                                 .format = GPU_SHADER_FORMAT,
                                                 .stage = SDL_GPU_SHADERSTAGE_FRAGMENT};
        SDL_GPUShader* const          vertex_shader =
            SDL_CreateGPUShader(vxray_instance.gpu_device, &vs_info);
        SDL_GPUShader* const fragment_shader =
            SDL_CreateGPUShader(vxray_instance.gpu_device, &ps_info);
        if (!vertex_shader || !fragment_shader)
        {
            SDL_LogError(SDL_LOG_CATEGORY_GPU, "Couldn't create brick-quad shaders: %s",
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

        vxray_instance.brick_quad_pipeline = SDL_CreateGPUGraphicsPipeline(
            vxray_instance.gpu_device,
            &(SDL_GPUGraphicsPipelineCreateInfo){
                .vertex_shader = vertex_shader,
                .fragment_shader = fragment_shader,
                .primitive_type = SDL_GPU_PRIMITIVETYPE_TRIANGLELIST,
                .rasterizer_state = (SDL_GPURasterizerState){.fill_mode = SDL_GPU_FILLMODE_FILL,
                                                             .cull_mode = SDL_GPU_CULLMODE_NONE,
                                                             .enable_depth_clip = true},
                .depth_stencil_state =
                    (SDL_GPUDepthStencilState){.compare_op = SDL_GPU_COMPAREOP_LESS,
                                               .enable_depth_test = true,
                                               .enable_depth_write = true},
                .target_info = (SDL_GPUGraphicsPipelineTargetInfo){
                    .num_color_targets = 1,
                    .color_target_descriptions =
                        (SDL_GPUColorTargetDescription[]){
                            {.format = SDL_GPU_TEXTUREFORMAT_R32_UINT}},
                    .depth_stencil_format = SDL_GPU_TEXTUREFORMAT_D32_FLOAT,
                    .has_depth_stencil_target = true}});
        SDL_ReleaseGPUShader(vxray_instance.gpu_device, fragment_shader);
        SDL_ReleaseGPUShader(vxray_instance.gpu_device, vertex_shader);
        if (!vxray_instance.brick_quad_pipeline)
        {
            SDL_LogError(SDL_LOG_CATEGORY_GPU, "Couldn't create brick-quad graphics pipeline: %s",
                         SDL_GetError());
            return SDL_APP_FAILURE;
        }
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
                                                 .num_storage_textures = 3,
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

        vxray_instance.gbuffer_pipeline = SDL_CreateGPUGraphicsPipeline(
            vxray_instance.gpu_device,
            &(SDL_GPUGraphicsPipelineCreateInfo){
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
                .target_info = (SDL_GPUGraphicsPipelineTargetInfo){
                    .num_color_targets = 2,
                    .color_target_descriptions =
                        (SDL_GPUColorTargetDescription[]){
                            {.format = SDL_GPU_TEXTUREFORMAT_R32_UINT},
                            {.format = SDL_GPU_TEXTUREFORMAT_R8_UINT}},
                    .depth_stencil_format = SDL_GPU_TEXTUREFORMAT_D32_FLOAT,
                    .has_depth_stencil_target = true}});
        SDL_ReleaseGPUShader(vxray_instance.gpu_device, fragment_shader);
        SDL_ReleaseGPUShader(vxray_instance.gpu_device, vertex_shader);
        if (!vxray_instance.gbuffer_pipeline)
        {
            SDL_LogError(SDL_LOG_CATEGORY_GPU, "Couldn't create G-buffer pipeline: %s",
                         SDL_GetError());
            return SDL_APP_FAILURE;
        }
    }

    // RTAO spatial-cache update pipeline

    vxray_instance.rtao_pipeline = SDL_CreateGPUComputePipeline(
        vxray_instance.gpu_device,
        &(SDL_GPUComputePipelineCreateInfo){.code_size = RTAO_CS_SIZE,
                                            .code = RTAO_CS_BYTES,
                                            .entrypoint = GPU_SHADER_ENTRYPOINT,
                                            .format = GPU_SHADER_FORMAT,
                                            .num_samplers = 1,
                                            .num_readonly_storage_textures = 2,
                                            .num_readwrite_storage_buffers = 3,
                                            .num_uniform_buffers = 1,
                                            .threadcount_x = 8,
                                            .threadcount_y = 8,
                                            .threadcount_z = 1});
    if (!vxray_instance.rtao_pipeline)
    {
        SDL_LogError(SDL_LOG_CATEGORY_GPU, "Couldn't create RTAO compute pipeline: %s",
                     SDL_GetError());
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
                                                 .num_samplers = 1,
                                                 .num_storage_textures = 3,
                                                 .num_storage_buffers = 2,
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

        vxray_instance.display_pipeline = SDL_CreateGPUGraphicsPipeline(
            vxray_instance.gpu_device,
            &(SDL_GPUGraphicsPipelineCreateInfo){
                .vertex_shader = vertex_shader,
                .fragment_shader = fragment_shader,
                .primitive_type = SDL_GPU_PRIMITIVETYPE_TRIANGLELIST,
                .rasterizer_state =
                    (SDL_GPURasterizerState){.fill_mode = SDL_GPU_FILLMODE_FILL,
                                             .cull_mode = SDL_GPU_CULLMODE_BACK,
                                             .front_face = SDL_GPU_FRONTFACE_CLOCKWISE},
                .target_info = (SDL_GPUGraphicsPipelineTargetInfo){
                    .num_color_targets = 1,
                    .color_target_descriptions = (SDL_GPUColorTargetDescription[]){
                        {.format = SDL_GetGPUSwapchainTextureFormat(vxray_instance.gpu_device,
                                                                    vxray_instance.window)}}}});
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
            vxray_instance.brick_grid_ext = scene.brick_grid_ext;

            uint32_t const brick_count = (uint32_t)scene.brick_grid.count;
            // Along each axis, at most every other brick in a row can emit a visible face.
            vxray_instance.face_capacity = brick_count == 1u ? 3u : 3u * brick_count / 2u;
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

            {
                uint32_t const        voxel_buffer_size = (uint32_t)scene.voxel_grid.count;
                SDL_GPUTexture* const voxel_texture = SDL_CreateGPUTexture(
                    device, &(SDL_GPUTextureCreateInfo){
                                .type = SDL_GPU_TEXTURETYPE_3D,
                                .format = SDL_GPU_TEXTUREFORMAT_R8_UINT,
                                .usage = SDL_GPU_TEXTUREUSAGE_GRAPHICS_STORAGE_READ |
                                         SDL_GPU_TEXTUREUSAGE_COMPUTE_STORAGE_READ,
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
                uint32_t const        brick_grid_size = (uint32_t)scene.brick_grid.count;
                SDL_GPUTexture* const brick_texture = SDL_CreateGPUTexture(
                    device, &(SDL_GPUTextureCreateInfo){
                                .type = SDL_GPU_TEXTURETYPE_3D,
                                .format = SDL_GPU_TEXTUREFORMAT_R8_UINT,
                                .usage = SDL_GPU_TEXTUREUSAGE_GRAPHICS_STORAGE_READ |
                                         SDL_GPU_TEXTUREUSAGE_COMPUTE_STORAGE_READ,
                                .width = scene.brick_grid_ext,
                                .height = scene.brick_grid_ext,
                                .layer_count_or_depth = scene.brick_grid_ext,
                                .num_levels = 1,
                                .sample_count = SDL_GPU_SAMPLECOUNT_1});
                if (!brick_texture)
                {
                    SDL_LogError(SDL_LOG_CATEGORY_GPU, "Failed to create brick grid texture: %s",
                                 SDL_GetError());
                    vx_scene_free(&scene);
                    return SDL_APP_FAILURE;
                }
                if (!vx_gpu_texture_upload(device, brick_texture, scene.brick_grid.ptr,
                                           brick_grid_size, (uint32_t)scene.brick_grid_ext))
                {
                    vx_scene_free(&scene);
                    return SDL_APP_FAILURE;
                }
                vxray_instance.brick_texture = brick_texture;
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
            {
                uint32_t const visible_faces_size =
                    vxray_instance.face_capacity * (uint32_t)sizeof(uint32_t);
                SDL_GPUBuffer* const visible_faces_buffer = SDL_CreateGPUBuffer(
                    device,
                    &(SDL_GPUBufferCreateInfo){.usage = SDL_GPU_BUFFERUSAGE_COMPUTE_STORAGE_WRITE |
                                                        SDL_GPU_BUFFERUSAGE_GRAPHICS_STORAGE_READ,
                                               .size = visible_faces_size});
                if (!visible_faces_buffer)
                {
                    SDL_LogError(SDL_LOG_CATEGORY_GPU, "Failed to create visible-face buffer: %s",
                                 SDL_GetError());
                    vx_scene_free(&scene);
                    return SDL_APP_FAILURE;
                }
                vxray_instance.visible_faces_buffer = visible_faces_buffer;
            }
            {
                SDL_GPUBuffer* const indirect_draw_buffer = SDL_CreateGPUBuffer(
                    device, &(SDL_GPUBufferCreateInfo){
                                .usage = SDL_GPU_BUFFERUSAGE_COMPUTE_STORAGE_WRITE |
                                         SDL_GPU_BUFFERUSAGE_INDIRECT,
                                .size = (uint32_t)sizeof(SDL_GPUIndirectDrawCommand)});
                if (!indirect_draw_buffer)
                {
                    SDL_LogError(SDL_LOG_CATEGORY_GPU, "Failed to create indirect draw buffer: %s",
                                 SDL_GetError());
                    vx_scene_free(&scene);
                    return SDL_APP_FAILURE;
                }
                vxray_instance.indirect_draw_buffer = indirect_draw_buffer;
            }
            {
                SDL_GPUTransferBuffer* const reset_transfer_buffer = SDL_CreateGPUTransferBuffer(
                    device, &(SDL_GPUTransferBufferCreateInfo){
                                .usage = SDL_GPU_TRANSFERBUFFERUSAGE_UPLOAD,
                                .size = (uint32_t)sizeof(SDL_GPUIndirectDrawCommand)});
                if (!reset_transfer_buffer)
                {
                    SDL_LogError(SDL_LOG_CATEGORY_GPU,
                                 "Failed to create indirect reset transfer buffer: %s",
                                 SDL_GetError());
                    vx_scene_free(&scene);
                    return SDL_APP_FAILURE;
                }
                vxray_instance.indirect_reset_transfer_buffer = reset_transfer_buffer;
            }
            {
                uint32_t const ao_buffer_size = (uint32_t)(VX_AO_HASH_CAPACITY * sizeof(uint32_t));
                SDL_GPUBufferCreateInfo const buffer_info = {
                    .usage = SDL_GPU_BUFFERUSAGE_COMPUTE_STORAGE_WRITE |
                             SDL_GPU_BUFFERUSAGE_GRAPHICS_STORAGE_READ,
                    .size = ao_buffer_size};
                SDL_GPUBuffer* const checksum_buffer = SDL_CreateGPUBuffer(device, &buffer_info);
                SDL_GPUBuffer* const payload_buffer = SDL_CreateGPUBuffer(device, &buffer_info);
                SDL_GPUBuffer* const last_touched_frame_buffer =
                    SDL_CreateGPUBuffer(device, &buffer_info);
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

                SDL_GPUTransferBuffer* const reset_transfer = SDL_CreateGPUTransferBuffer(
                    device,
                    &(SDL_GPUTransferBufferCreateInfo){.usage = SDL_GPU_TRANSFERBUFFERUSAGE_UPLOAD,
                                                       .size = ao_buffer_size});
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
    igRadioButton_IntPtr("Brick coordinates", &vxray_instance.display_texture,
                         VX_DISPLAY_TEXTURE_BRICK_COORDINATES);
    igRadioButton_IntPtr("Ambient visibility", &vxray_instance.display_texture,
                         VX_DISPLAY_TEXTURE_AMBIENT_VISIBILITY);
    igRadioButton_IntPtr("Cell size", &vxray_instance.display_texture,
                         VX_DISPLAY_TEXTURE_CELL_SIZE);
    igSliderFloat("sp", &vxray_instance.ao_sp, 1.f, 16.f, "%.1f px", 0);
    igSliderFloat("smin", &vxray_instance.ao_smin, 0.01f, 2.f, "%.3f",
                  ImGuiSliderFlags_Logarithmic);
    igText("Face capacity: %u", vxray_instance.face_capacity);
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
    assert(isfinite(vxray_instance.ao_sp) && vxray_instance.ao_sp > 0.f);
    assert(isfinite(vxray_instance.ao_smin) && vxray_instance.ao_smin > 0.f);

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
    mat4s const view = glms_look_lh_zo(camera->position, forward, world_up);
    mat4s       projection = glms_perspective_lh_zo(fov, aspect, near_plane, far_plane);
    mat4s const view_projection = glms_mat4_mul(projection, view);
    mat4s const inverse_view_projection = glms_mat4_inv(view_projection);
    brick_quad_uniforms const brick_uniforms = {
        .camera_position = vx_float4_from_vec3(camera->position, 0.f),
        .view_projection = vx_float4x4_from_mat4(view_projection),
        .brick_grid_ext = (uint)vxray_instance.brick_grid_ext};
    gbuffer_uniforms const gbuffer_uniform_data = {
        .camera_pos = vx_float4_from_vec3(camera->position, 0.f),
        .inverse_view_projection = vx_float4x4_from_mat4(inverse_view_projection),
        .view_projection = vx_float4x4_from_mat4(view_projection),
        .grid_ext = vxray_instance.grid_ext};
    rtao_uniforms const rtao_uniform_data = {
        .camera_pos = vx_float4_from_vec3(camera->position, 0.f),
        .inverse_view_projection = vx_float4x4_from_mat4(inverse_view_projection),
        .rtao_radius = (float)VX_BRICK_EXT,
        .sp = vxray_instance.ao_sp,
        .smin = vxray_instance.ao_smin,
        .vertical_fov = fov,
        .near_plane = near_plane,
        .far_plane = far_plane,
        .grid_ext = vxray_instance.grid_ext,
        .frame_index = vxray_instance.frame_index,
        .render_height = height};

    // Reset the complete indirect draw command on the GPU timeline.

    SDL_GPUIndirectDrawCommand const indirect_command = {
        .num_vertices = 6,
        .num_instances = 0,
        .first_vertex = 0,
        .first_instance = 0,
    };
    void* const reset_data =
        SDL_MapGPUTransferBuffer(gpu_device, vxray_instance.indirect_reset_transfer_buffer, true);
    if (!reset_data)
    {
        SDL_LogError(SDL_LOG_CATEGORY_GPU, "Couldn't map indirect reset buffer: %s",
                     SDL_GetError());
        SDL_CancelGPUCommandBuffer(cmd_buffer);
        return SDL_APP_FAILURE;
    }
    SDL_memcpy(reset_data, &indirect_command, sizeof(indirect_command));
    SDL_UnmapGPUTransferBuffer(gpu_device, vxray_instance.indirect_reset_transfer_buffer);

    SDL_GPUCopyPass* const copy_pass = SDL_BeginGPUCopyPass(cmd_buffer);
    assert(copy_pass);
    SDL_UploadToGPUBuffer(
        copy_pass,
        &(SDL_GPUTransferBufferLocation){
            .transfer_buffer = vxray_instance.indirect_reset_transfer_buffer, .offset = 0},
        &(SDL_GPUBufferRegion){.buffer = vxray_instance.indirect_draw_buffer,
                               .offset = 0,
                               .size = (uint32_t)sizeof(indirect_command)},
        true);
    if (vxray_instance.frame_index == 0u)
    {
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
    }
    SDL_EndGPUCopyPass(copy_pass);

    // Generate exposed, camera-facing brick faces.

    SDL_GPUStorageBufferReadWriteBinding const writable_buffers[] = {
        {.buffer = vxray_instance.visible_faces_buffer, .cycle = true},
        {.buffer = vxray_instance.indirect_draw_buffer, .cycle = false},
    };
    SDL_GPUComputePass* const compute_pass = SDL_BeginGPUComputePass(
        cmd_buffer, 0, 0, writable_buffers, SDL_arraysize(writable_buffers));
    assert(compute_pass);
    SDL_BindGPUComputePipeline(compute_pass, vxray_instance.brick_quad_compute_pipeline);
    SDL_GPUTexture* const compute_textures[] = {vxray_instance.brick_texture};
    SDL_BindGPUComputeStorageTextures(compute_pass, 0, compute_textures,
                                      SDL_arraysize(compute_textures));
    SDL_PushGPUComputeUniformData(cmd_buffer, 0, &brick_uniforms, sizeof(brick_uniforms));
    uint32_t const brick_count =
        (uint32_t)(vxray_instance.brick_grid_ext * vxray_instance.brick_grid_ext *
                   vxray_instance.brick_grid_ext);
    SDL_DispatchGPUCompute(compute_pass, (brick_count + 63u) / 64u, 1, 1);
    SDL_EndGPUComputePass(compute_pass);

    // Rasterize the generated quads into entry depth and packed brick coordinates.

    SDL_GPUColorTargetInfo const brick_target_info = {.texture = vxray_instance.entry_brick_texture,
                                                      .clear_color =
                                                          (SDL_FColor){0.f, 0.f, 0.f, 0.f},
                                                      .load_op = SDL_GPU_LOADOP_CLEAR,
                                                      .store_op = SDL_GPU_STOREOP_STORE,
                                                      .cycle = true};

    SDL_GPUDepthStencilTargetInfo const depth_target_info = {
        .texture = vxray_instance.entry_depth_texture,
        .clear_depth = 1.f,
        .load_op = SDL_GPU_LOADOP_CLEAR,
        .store_op = SDL_GPU_STOREOP_DONT_CARE,
        .stencil_load_op = SDL_GPU_LOADOP_DONT_CARE,
        .stencil_store_op = SDL_GPU_STOREOP_DONT_CARE,
        .cycle = true};
    SDL_GPURenderPass* const brick_pass =
        SDL_BeginGPURenderPass(cmd_buffer, &brick_target_info, 1, &depth_target_info);
    assert(brick_pass);
    SDL_BindGPUGraphicsPipeline(brick_pass, vxray_instance.brick_quad_pipeline);
    SDL_GPUBuffer* const vertex_storage_buffers[] = {vxray_instance.visible_faces_buffer};
    SDL_BindGPUVertexStorageBuffers(brick_pass, 0, vertex_storage_buffers,
                                    SDL_arraysize(vertex_storage_buffers));
    SDL_PushGPUVertexUniformData(cmd_buffer, 0, &brick_uniforms, sizeof(brick_uniforms));
    SDL_DrawGPUPrimitivesIndirect(brick_pass, vxray_instance.indirect_draw_buffer, 0, 1);
    SDL_EndGPURenderPass(brick_pass);

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
        vxray_instance.entry_brick_texture,
        vxray_instance.voxel_texture,
        vxray_instance.brick_texture,
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

    // Update the persistent ambient-visibility spatial cache.

    SDL_GPUStorageBufferReadWriteBinding const rtao_writable_buffers[] = {
        {.buffer = vxray_instance.ao_checksum_buffer, .cycle = false},
        {.buffer = vxray_instance.ao_payload_buffer, .cycle = false},
        {.buffer = vxray_instance.ao_last_touched_frame_buffer, .cycle = false},
    };
    SDL_GPUComputePass* const rtao_pass = SDL_BeginGPUComputePass(
        cmd_buffer, 0, 0, rtao_writable_buffers, SDL_arraysize(rtao_writable_buffers));
    assert(rtao_pass);
    SDL_BindGPUComputePipeline(rtao_pass, vxray_instance.rtao_pipeline);
    SDL_GPUTextureSamplerBinding const rtao_depth_binding = {
        .texture = vxray_instance.gbuffer_depth_texture, .sampler = vxray_instance.display_sampler};
    SDL_BindGPUComputeSamplers(rtao_pass, 0, &rtao_depth_binding, 1);
    SDL_GPUTexture* const rtao_storage_textures[] = {
        vxray_instance.gbuffer_normal_texture,
        vxray_instance.voxel_texture,
    };
    SDL_BindGPUComputeStorageTextures(rtao_pass, 0, rtao_storage_textures,
                                      SDL_arraysize(rtao_storage_textures));
    SDL_PushGPUComputeUniformData(cmd_buffer, 0, &rtao_uniform_data, sizeof(rtao_uniform_data));
    uint32_t const rtao_width = (width + 1u) / 2u;
    SDL_DispatchGPUCompute(rtao_pass, (rtao_width + 7u) / 8u, (height + 7u) / 8u, 1);
    SDL_EndGPUComputePass(rtao_pass);

    // Display the selected intermediate texture on the swapchain.

    SDL_GPUColorTargetInfo const color_target_info = {.texture = swapchain_texture,
                                                      .clear_color =
                                                          (SDL_FColor){0.f, 0.f, 0.f, 0.f},
                                                      .load_op = SDL_GPU_LOADOP_CLEAR,
                                                      .store_op = SDL_GPU_STOREOP_STORE};
    SDL_GPURenderPass* render_pass = SDL_BeginGPURenderPass(cmd_buffer, &color_target_info, 1, 0);
    assert(render_pass);

    SDL_BindGPUGraphicsPipeline(render_pass, vxray_instance.display_pipeline);
    SDL_GPUTextureSamplerBinding const display_binding = {
        .texture = vxray_instance.gbuffer_depth_texture, .sampler = vxray_instance.display_sampler};
    SDL_BindGPUFragmentSamplers(render_pass, 0, &display_binding, 1);
    SDL_GPUTexture* const display_storage_textures[] = {
        vxray_instance.entry_brick_texture,
        vxray_instance.gbuffer_albedo_texture,
        vxray_instance.gbuffer_normal_texture,
    };
    SDL_BindGPUFragmentStorageTextures(render_pass, 0, display_storage_textures,
                                       SDL_arraysize(display_storage_textures));
    SDL_GPUBuffer* const display_storage_buffers[] = {
        vxray_instance.ao_checksum_buffer,
        vxray_instance.ao_payload_buffer,
    };
    SDL_BindGPUFragmentStorageBuffers(render_pass, 0, display_storage_buffers,
                                      SDL_arraysize(display_storage_buffers));
    display_uniforms const display_uniform_data = {
        .inverse_view_projection = vx_float4x4_from_mat4(inverse_view_projection),
        .texture_type = (uint)vxray_instance.display_texture,
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

    if (vxray_instance.indirect_reset_transfer_buffer)
    {
        SDL_ReleaseGPUTransferBuffer(vxray_instance.gpu_device,
                                     vxray_instance.indirect_reset_transfer_buffer);
        vxray_instance.indirect_reset_transfer_buffer = 0;
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

    if (vxray_instance.indirect_draw_buffer)
    {
        SDL_ReleaseGPUBuffer(vxray_instance.gpu_device, vxray_instance.indirect_draw_buffer);
        vxray_instance.indirect_draw_buffer = 0;
    }

    if (vxray_instance.visible_faces_buffer)
    {
        SDL_ReleaseGPUBuffer(vxray_instance.gpu_device, vxray_instance.visible_faces_buffer);
        vxray_instance.visible_faces_buffer = 0;
    }

    if (vxray_instance.palette_buffer)
    {
        SDL_ReleaseGPUBuffer(vxray_instance.gpu_device, vxray_instance.palette_buffer);
        vxray_instance.palette_buffer = 0;
    }

    if (vxray_instance.brick_texture)
    {
        SDL_ReleaseGPUTexture(vxray_instance.gpu_device, vxray_instance.brick_texture);
        vxray_instance.brick_texture = 0;
    }

    if (vxray_instance.voxel_texture)
    {
        SDL_ReleaseGPUTexture(vxray_instance.gpu_device, vxray_instance.voxel_texture);
        vxray_instance.voxel_texture = 0;
    }

    if (vxray_instance.entry_depth_texture)
    {
        SDL_ReleaseGPUTexture(vxray_instance.gpu_device, vxray_instance.entry_depth_texture);
        vxray_instance.entry_depth_texture = 0;
    }

    if (vxray_instance.entry_brick_texture)
    {
        SDL_ReleaseGPUTexture(vxray_instance.gpu_device, vxray_instance.entry_brick_texture);
        vxray_instance.entry_brick_texture = 0;
    }

    if (vxray_instance.gbuffer_depth_texture)
    {
        SDL_ReleaseGPUTexture(vxray_instance.gpu_device, vxray_instance.gbuffer_depth_texture);
        vxray_instance.gbuffer_depth_texture = 0;
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

    if (vxray_instance.gbuffer_pipeline)
    {
        SDL_ReleaseGPUGraphicsPipeline(vxray_instance.gpu_device, vxray_instance.gbuffer_pipeline);
        vxray_instance.gbuffer_pipeline = 0;
    }

    if (vxray_instance.rtao_pipeline)
    {
        SDL_ReleaseGPUComputePipeline(vxray_instance.gpu_device, vxray_instance.rtao_pipeline);
        vxray_instance.rtao_pipeline = 0;
    }

    if (vxray_instance.display_pipeline)
    {
        SDL_ReleaseGPUGraphicsPipeline(vxray_instance.gpu_device, vxray_instance.display_pipeline);
        vxray_instance.display_pipeline = 0;
    }

    if (vxray_instance.brick_quad_pipeline)
    {
        SDL_ReleaseGPUGraphicsPipeline(vxray_instance.gpu_device,
                                       vxray_instance.brick_quad_pipeline);
        vxray_instance.brick_quad_pipeline = 0;
    }

    if (vxray_instance.brick_quad_compute_pipeline)
    {
        SDL_ReleaseGPUComputePipeline(vxray_instance.gpu_device,
                                      vxray_instance.brick_quad_compute_pipeline);
        vxray_instance.brick_quad_compute_pipeline = 0;
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
