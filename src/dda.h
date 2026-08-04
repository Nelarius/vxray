#pragma once

#include "hlsl_shim.h"

typedef struct dda_uniforms
{
    float4 camera_pos;
    float4 inverse_view_projection_0;
    float4 inverse_view_projection_1;
    float4 inverse_view_projection_2;
    float4 inverse_view_projection_3;
    int    grid_ext;
    int    pad0;
    int    pad1;
    int    pad2;
} dda_uniforms;
