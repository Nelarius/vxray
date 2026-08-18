#pragma once

#include "hlsl_shim.h"

#define VX_AO_HASH_CAPACITY (1u << 23u)
#define VX_AO_HASH_MASK (VX_AO_HASH_CAPACITY - 1u)
#define VX_AO_HASH_PROBE_COUNT 16u
#define VX_AO_MAX_CELL_AGE 20u
#define VX_AO_SAMPLE_LIMIT 500u
#define VX_AO_RAYS_PER_PIXEL 2u

typedef struct rtao_uniforms
{
    float4   camera_pos;
    float4x4 inverse_view_projection;
    float    rtao_radius;
    float    sp;
    float    smin;
    float    vertical_fov;
    float    near_plane;
    float    far_plane;
    int      grid_ext;
    uint     frame_index;
    uint     render_height;
    uint     pad0;
    uint     pad1;
    uint     pad2;
} rtao_uniforms;

typedef struct rtao_filter_uniforms
{
    float4x4 view_matrix;
    float    sigma_depth;
    uint     step_width;
    float    near_plane;
    float    far_plane;
} rtao_filter_uniforms;
