#pragma once

#include "hlsl_shim.h"

#define VX_PATH_TRACE_SAMPLE_LIMIT 4000u
#define VX_PATH_TRACE_BOUNCE_COUNT 4u
#define VX_PATH_TRACE_ACCUMULATION_SCALE 4096u
#define VX_PATH_TRACE_MAX_SAMPLE_SHADING 255u
#define VX_PATH_TRACE_SPATIAL_HASH_CAPACITY (1u << 24u)
#define VX_PATH_TRACE_SPATIAL_HASH_MASK (VX_PATH_TRACE_SPATIAL_HASH_CAPACITY - 1u)
#define VX_PATH_TRACE_SPATIAL_HASH_PROBE_COUNT 16u
#define VX_PATH_TRACE_SPATIAL_HASH_MAX_CELL_AGE 20u
#define VX_PATH_TRACE_SPATIAL_HASH_TOUCH_PERIOD 16u
#define VX_WAVEFRONT_SCREEN_THREAD_COUNT 8u
#define VX_WAVEFRONT_EXTEND_THREAD_COUNT 64u

typedef struct path_tracer_index_uniforms
{
    float4x4 inverse_view_projection;
    float4x4 previous_view_projection;
    float    sp;
    float    smin;
    float    vertical_fov;
    float    near_plane;
    float    far_plane;
    uint     frame;
    uint     render_height;
    uint     history_valid;
} path_tracer_index_uniforms;

typedef struct path_tracer_uniforms
{
    float4   camera_pos;
    float4x4 inverse_view_projection;
    float4   sun_direction;
    float4   transmitted_sun_color;
    int      grid_ext;
    uint     frame;
    uint     bounce;
    uint     pad1;
} path_tracer_uniforms;

typedef struct path_tracer_ray
{
    float4 origin_and_path_index;
    float3 direction;
} path_tracer_ray;

typedef struct path_tracer_path_state
{
    float4 throughput_and_spatial_index;
    float4 radiance;
} path_tracer_path_state;
