#pragma once

#include "hlsl_shim.h"

#define VX_DISPLAY_TEXTURE_ALBEDO 0
#define VX_DISPLAY_TEXTURE_NORMAL 1
#define VX_DISPLAY_TEXTURE_SURFACE_DEPTH 2
#define VX_DISPLAY_TEXTURE_BRICK_AABB_DEPTH 3

typedef struct display_uniforms
{
    uint  texture_type;
    float near_plane;
    float far_plane;
    float visualization_range;
} display_uniforms;
