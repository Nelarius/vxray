#pragma once

#include "constants.h"
#include "hlsl_shim.h"

typedef struct brick_quad_uniforms
{
    float4   camera_position;
    float4x4 view_projection;
    uint     brick_grid_ext;
    uint     pad0;
    uint     pad1;
    uint     pad2;
} brick_quad_uniforms;

typedef struct depth_visualize_uniforms
{
    float near_plane;
    float far_plane;
    float visualization_range;
    float pad;
} depth_visualize_uniforms;
