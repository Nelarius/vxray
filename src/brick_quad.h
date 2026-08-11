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
