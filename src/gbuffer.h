#pragma once

#include "hlsl_shim.h"

typedef struct gbuffer_uniforms
{
    float4   camera_pos;
    float4x4 inverse_view_projection;
    float4x4 view_projection;
    int      grid_ext;
    uint     use_brick_prepass;
    int      pad1;
    int      pad2;
} gbuffer_uniforms;
