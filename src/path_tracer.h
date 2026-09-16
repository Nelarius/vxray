#pragma once

#include "hlsl_shim.h"

#define VX_PATH_TRACE_UPDATE_TILE_SIZE 4u
#define VX_PATH_TRACE_BOUNCE_COUNT 4u
#define VX_PATH_TRACE_ACCUMULATION_SCALE 4096u
#define VX_PATH_TRACE_MAX_SAMPLE_SHADING 255u
#define VX_PATH_TRACE_SPATIAL_HASH_CAPACITY (1u << 20u)
#define VX_PATH_TRACE_SPATIAL_HASH_MASK (VX_PATH_TRACE_SPATIAL_HASH_CAPACITY - 1u)
#define VX_PATH_TRACE_SPATIAL_HASH_PROBE_COUNT 16u
#define VX_PATH_TRACE_SPATIAL_HASH_MAX_CELL_AGE 128u
#define VX_SHARC_PROPAGATION_DEPTH 4u
#define VX_SHARC_HISTORY_SAMPLE_COUNT 64u
#define VX_SHARC_QUERY_MIN_SEGMENT_CELL_RATIO 1.0
#define VX_PATH_TRACE_MODE_UPDATE 0u
#define VX_PATH_TRACE_MODE_QUERY 1u
#define VX_WAVEFRONT_SCREEN_THREAD_COUNT 8u
#define VX_WAVEFRONT_EXTEND_THREAD_COUNT 64u

typedef struct path_tracer_uniforms
{
    float4   camera_pos;
    float4x4 inverse_view_projection;
    float4   sun_direction;
    float4   transmitted_sun_color;
    int      grid_ext;
    uint     frame;
    uint     bounce;
    uint     mode;
    float    sp;
    float    smin;
    float    vertical_fov;
    uint     render_height;
} path_tracer_uniforms;

typedef struct path_tracer_ray
{
    float4 origin_and_path_index;
    float3 direction;
} path_tracer_ray;

typedef struct path_tracer_path_state
{
    float4 throughput_and_path_length;
    float4 radiance;
    float4 sharc_vertices[VX_SHARC_PROPAGATION_DEPTH];
} path_tracer_path_state;
