#pragma once

#include "hlsl_shim.h"

#define VX_DISPLAY_TEXTURE_ALBEDO 0
#define VX_DISPLAY_TEXTURE_NORMAL 1
#define VX_DISPLAY_TEXTURE_AMBIENT_VISIBILITY 2
#define VX_DISPLAY_TEXTURE_CELL_SIZE 3
#define VX_DISPLAY_TEXTURE_SPATIAL_INDEX 4
#define VX_DISPLAY_TEXTURE_SKY_VIEW 5
#define VX_DISPLAY_TEXTURE_PATH_TRACE 6

typedef struct display_uniforms
{
    uint  texture_type;
    float near_plane;
    float far_plane;
    uint  grid_ext;
    float sp;
    float smin;
    float vertical_fov;
    uint  render_height;
    float exposure;
    uint  pad1;
    uint  pad2;
    uint  pad3;
} display_uniforms;
