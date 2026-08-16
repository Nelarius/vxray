#include "constants.h"
#include "display.h"

#include "spatial_hash.hlsli"

struct ps_input
{
    float4 position : SV_Position;
};

Texture2D<float> depth_tex : register(t0, space2);
Texture2D<float> visibility_tex : register(t1, space2);
Texture2D<uint>  entry_bricks : register(t2, space2);
Texture2D<uint>  albedo_tex : register(t3, space2);
Texture2D<uint>  normal_tex : register(t4, space2);

ConstantBuffer<display_uniforms> uniforms : register(b0, space3);

static float4 const BACKGROUND_COLOR = float4(0.02, 0.025, 0.03, 1.0);
static float4 const CACHE_FAILURE_COLOR = float4(1.0, 0.0, 1.0, 1.0);

float4 unpack_albedo(uint const rgba)
{
    float const r = (float)(rgba & 255u) / 255.0;
    float const g = (float)((rgba >> 8u) & 255u) / 255.0;
    float const b = (float)((rgba >> 16u) & 255u) / 255.0;
    float const a = (float)((rgba >> 24u) & 255u) / 255.0;
    return float4(pow(float3(r, g, b), 2.2), a);
}

float4 visualize_depth(float const depth)
{
    if (depth >= 1.0)
    {
        return BACKGROUND_COLOR;
    }

    float const linear_depth = linear_view_depth(depth, uniforms.near_plane, uniforms.far_plane);
    float const shade = 1.0 - saturate(linear_depth / (2.0 * (float)uniforms.grid_ext));
    return float4(shade, shade, shade, 1.0);
}

float4 visualize_brick_coordinates(uint const entry_brick_record)
{
    if (entry_brick_record == 0u)
    {
        return BACKGROUND_COLOR;
    }

    uint const   packed = entry_brick_record - 1u;
    uint3 const  brick = uint3(packed & 255u, (packed >> 8u) & 255u, (packed >> 16u) & 255u);
    float const  brick_grid_ext = (float)(uniforms.grid_ext / VX_BRICK_EXT);
    float3 const normalized_brick = (float3(brick) + 0.5) / brick_grid_ext;
    return float4(0.15 + 0.85 * normalized_brick, 1.0);
}

float3 cell_size_color(float const cell_size, float const smin)
{
    // Adjacent powers of two receive distinct colors, including sizes below smin.
    float3 const palette[8] = {
        float3(0.1216, 0.4667, 0.7059), float3(1.0000, 0.4980, 0.0549),
        float3(0.1725, 0.6275, 0.1725), float3(0.8392, 0.1529, 0.1569),
        float3(0.5804, 0.4039, 0.7412), float3(0.5490, 0.3373, 0.2941),
        float3(0.8902, 0.4667, 0.7608), float3(0.4980, 0.4980, 0.4980),
    };
    int const exponent = (int)round(log2(cell_size / smin));
    return palette[(uint)exponent & 7u];
}

float4 main(ps_input const input) : SV_Target0
{
    uint width, height;
    depth_tex.GetDimensions(width, height);
    uint2 const pixel = min(uint2(input.position.xy), uint2(width - 1u, height - 1u));
    float const surface_depth = depth_tex.Load(int3(pixel, 0)).r;

    if (uniforms.texture_type == VX_DISPLAY_TEXTURE_ALBEDO ||
        uniforms.texture_type == VX_DISPLAY_TEXTURE_AMBIENT_VISIBILITY ||
        uniforms.texture_type == VX_DISPLAY_TEXTURE_NORMAL ||
        uniforms.texture_type == VX_DISPLAY_TEXTURE_CELL_SIZE)
    {
        if (surface_depth >= 1.0)
        {
            return BACKGROUND_COLOR;
        }

        int3 const texel = int3(pixel, 0);

        if (uniforms.texture_type == VX_DISPLAY_TEXTURE_NORMAL)
        {
            uint const packed_normal = normal_tex.Load(texel).r;
            return float4(unpack_normal(packed_normal) * 0.5 + 0.5, 1.0);
        }
        if (uniforms.texture_type == VX_DISPLAY_TEXTURE_CELL_SIZE)
        {
            float const view_depth =
                linear_view_depth(surface_depth, uniforms.near_plane, uniforms.far_plane);
            float const cell_size =
                compute_cell_size(view_depth, uniforms.vertical_fov, uniforms.render_height,
                                  uniforms.sp, uniforms.smin);
            return float4(cell_size_color(cell_size, uniforms.smin), 1.0);
        }

        float const visibility = visibility_tex.Load(texel).r;
        if (visibility < 0.0)
        {
            return CACHE_FAILURE_COLOR;
        }
        if (uniforms.texture_type == VX_DISPLAY_TEXTURE_AMBIENT_VISIBILITY)
        {
            return float4(visibility, visibility, visibility, 1.0);
        }

        float4 const color = unpack_albedo(albedo_tex.Load(texel).r);
        return float4(color.rgb * visibility, color.a);
    }

    if (uniforms.texture_type == VX_DISPLAY_TEXTURE_SURFACE_DEPTH)
    {
        return visualize_depth(surface_depth);
    }

    return visualize_brick_coordinates(entry_bricks.Load(int3(pixel, 0)).r);
}
