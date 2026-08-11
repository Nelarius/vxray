#include "display.h"

struct ps_input
{
    float2 uv : TEXCOORD0;
};

Texture2D<float> surface_depth_texture : register(t0, space2);
Texture2D<float> entry_depth_texture : register(t1, space2);
Texture2D<uint>  albedo_texture : register(t2, space2);
Texture2D<uint>  normal_texture : register(t3, space2);

SamplerState surface_depth_sampler : register(s0, space2);
SamplerState entry_depth_sampler : register(s1, space2);

ConstantBuffer<display_uniforms> uniforms : register(b0, space3);

static float4 const BACKGROUND_COLOR = float4(0.02, 0.025, 0.03, 1.0);

float4 unpack_albedo(uint const rgba)
{
    float const r = (float)(rgba & 255u) / 255.0;
    float const g = (float)((rgba >> 8u) & 255u) / 255.0;
    float const b = (float)((rgba >> 16u) & 255u) / 255.0;
    float const a = (float)((rgba >> 24u) & 255u) / 255.0;
    return float4(pow(float3(r, g, b), 2.2), a);
}

float3 unpack_normal(uint const packed)
{
    float3 const magnitude = float3(packed & 1u, (packed >> 2u) & 1u, (packed >> 4u) & 1u);
    float3 const negative = float3((packed >> 1u) & 1u, (packed >> 3u) & 1u, (packed >> 5u) & 1u);
    return magnitude * (1.0 - 2.0 * negative);
}

float4 visualize_depth(float const depth)
{
    if (depth >= 1.0)
    {
        return BACKGROUND_COLOR;
    }

    float const linear_depth =
        uniforms.near_plane * uniforms.far_plane /
        (uniforms.far_plane - depth * (uniforms.far_plane - uniforms.near_plane));
    float const shade = 1.0 - saturate(linear_depth / uniforms.visualization_range);
    return float4(shade, shade, shade, 1.0);
}

float4 main(ps_input const input) : SV_Target0
{
    if (uniforms.texture_type == VX_DISPLAY_TEXTURE_ALBEDO)
    {
        float const depth =
            surface_depth_texture.SampleLevel(surface_depth_sampler, input.uv, 0.0).r;
        if (depth >= 1.0)
        {
            return BACKGROUND_COLOR;
        }
        uint texture_width;
        uint texture_height;
        albedo_texture.GetDimensions(texture_width, texture_height);
        int3 const texel = int3(input.uv * uint2(texture_width, texture_height), 0);
        uint const albedo = albedo_texture.Load(texel).r;
        return unpack_albedo(albedo);
    }
    if (uniforms.texture_type == VX_DISPLAY_TEXTURE_NORMAL)
    {
        float const depth =
            surface_depth_texture.SampleLevel(surface_depth_sampler, input.uv, 0.0).r;
        if (depth >= 1.0)
        {
            return BACKGROUND_COLOR;
        }
        uint texture_width;
        uint texture_height;
        normal_texture.GetDimensions(texture_width, texture_height);
        int3 const texel = int3(input.uv * uint2(texture_width, texture_height), 0);
        uint const packed = normal_texture.Load(texel).r;
        return float4(unpack_normal(packed) * 0.5 + 0.5, 1.0);
    }
    if (uniforms.texture_type == VX_DISPLAY_TEXTURE_SURFACE_DEPTH)
    {
        float const depth =
            surface_depth_texture.SampleLevel(surface_depth_sampler, input.uv, 0.0).r;
        return visualize_depth(depth);
    }

    float const depth = entry_depth_texture.SampleLevel(entry_depth_sampler, input.uv, 0.0).r;
    return visualize_depth(depth);
}
