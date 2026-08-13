#pragma once

#include "hlsl_shim.h"

typedef struct rtao_uniforms
{
    float4   camera_pos;
    float4x4 inverse_view_projection;
    float    rtao_radius;
    int      grid_ext;
    int      pad0;
    int      pad1;
} rtao_uniforms;
