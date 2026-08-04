#include "brick_quad.h"

struct ps_input
{
    float2 uv : TEXCOORD0;
};

Texture2D<float> entry_depth : register(t0, space2);
SamplerState     entry_sampler : register(s0, space2);

ConstantBuffer<depth_visualize_uniforms> uniforms : register(b0, space3);

float4 main(ps_input const input) : SV_Target0
{
    float const depth = entry_depth.SampleLevel(entry_sampler, input.uv, 0.0).r;
    if (depth >= 1.0)
    {
        return float4(0.02, 0.025, 0.03, 1.0);
    }

    float const linear_depth =
        uniforms.near_plane * uniforms.far_plane /
        (uniforms.far_plane - depth * (uniforms.far_plane - uniforms.near_plane));
    float const shade = 1.0 - saturate(linear_depth / uniforms.visualization_range);
    return float4(shade, shade, shade, 1.0);
}
