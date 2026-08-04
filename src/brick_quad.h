#pragma once

#include "constants.h"
#include "hlsl_shim.h"

typedef struct brick_quad_uniforms
{
    float4 camera_position;
    float4 camera_right;
    float4 camera_up;
    float4 camera_forward;
    float2 projection_scale;
    float  near_plane;
    float  far_plane;
    uint   brick_grid_ext;
    uint   face_capacity;
    uint   pad0;
    uint   pad1;
} brick_quad_uniforms;

typedef struct depth_visualize_uniforms
{
    float near_plane;
    float far_plane;
    float visualization_range;
    float pad;
} depth_visualize_uniforms;
