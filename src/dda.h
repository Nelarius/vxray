#pragma once

#include "hlsl_shim.h"

typedef struct dda_uniforms
{
    float4 camera_pos;
    float4 camera_right;
    float4 camera_up;
    float4 camera_forward;
    float4 viewport;
    int    grid_ext;
} dda_uniforms;
