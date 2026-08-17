#pragma once

#include "hlsl_shim.h"

#define VX_DISPLAY_TEXTURE_ALBEDO 0
#define VX_DISPLAY_TEXTURE_NORMAL 1
#define VX_DISPLAY_TEXTURE_SURFACE_DEPTH 2
#define VX_DISPLAY_TEXTURE_BRICK_COORDINATES 3
#define VX_DISPLAY_TEXTURE_AMBIENT_VISIBILITY 4
#define VX_DISPLAY_TEXTURE_CELL_SIZE 5

typedef struct display_uniforms
{
    float4x4 inverse_view_projection;
    uint     texture_type;
    float    near_plane;
    float    far_plane;
    uint     grid_ext;
    float    sp;
    float    smin;
    float    vertical_fov;
    uint     render_height;
} display_uniforms;
