#pragma once

#include "hlsl_shim.h"

typedef struct dda_uniforms
{
    float4   camera_pos;
    float4x4 inverse_view_projection;
    int      grid_ext;
    int      pad0;
    int      pad1;
    int      pad2;
} dda_uniforms;
