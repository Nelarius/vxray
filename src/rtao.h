#pragma once

#include "hlsl_shim.h"

#define VX_AO_HASH_CAPACITY (1u << 23u)
#define VX_AO_HASH_MASK (VX_AO_HASH_CAPACITY - 1u)
#define VX_AO_HASH_PROBE_COUNT 16u
#define VX_AO_MAX_CELL_AGE 20u
#define VX_AO_HASH_TOUCH_PERIOD 16u
#define VX_AO_SAMPLE_LIMIT 2000u

typedef struct rtao_uniforms
{
    float4   camera_pos;
    float4x4 inverse_view_projection;
    float4x4 previous_view_projection;
    float    rtao_radius;
    float    sp;
    float    smin;
    float    vertical_fov;
    float    near_plane;
    float    far_plane;
    int      grid_ext;
    uint     frame_index;
    uint     render_height;
    uint     sample_index;
    uint     history_valid;
    uint     pad2;
} rtao_uniforms;
