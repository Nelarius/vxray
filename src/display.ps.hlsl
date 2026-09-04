#include "constants.h"
#include "display.h"
#include "path_tracer.h"

#include "spatial_hash.hlsli"

struct ps_input
{
    float4 position : SV_Position;
};

Texture2D<float>        depth_tex : register(t0, space2);
Texture2D<float>        visibility_tex : register(t1, space2);
Texture2D<float4>       sky_view_tex : register(t2, space2);
Texture2D<uint>         albedo_tex : register(t3, space2);
Texture2D<uint>         normal_tex : register(t4, space2);
Texture2D<uint>         spatial_index_tex : register(t5, space2);
StructuredBuffer<uint4> path_trace_payloads : register(t6, space2);

SamplerState depth_sampler : register(s0, space2);
SamplerState visibility_sampler : register(s1, space2);
SamplerState sky_view_sampler : register(s2, space2);

ConstantBuffer<display_uniforms> uniforms : register(b0, space3);

static float4 const BACKGROUND_COLOR = float4(0.02, 0.025, 0.03, 1.0);
static float4 const CACHE_FAILURE_COLOR = float4(1.0, 0.0, 1.0, 1.0);

float3 aces_filmic(float3 const x)
{
    float const a = 2.51;
    float const b = 0.03;
    float const c = 2.43;
    float const d = 0.59;
    float const e = 0.14;
    return saturate((x * (a * x + b)) / (x * (c * x + d) + e));
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

float3 spatial_index_color(uint const index)
{
    uint const color = pcg(index);
    return 0.25 + 0.75 * float3(color & 255u, (color >> 8u) & 255u, (color >> 16u) & 255u) / 255.0;
}

float4 main(ps_input const input) : SV_Target0
{
    uint width, height;
    depth_tex.GetDimensions(width, height);
    uint2 const  pixel = min(uint2(input.position.xy), uint2(width - 1u, height - 1u));
    float2 const pixel_uv = (float2(pixel) + 0.5) / float2(width, height);
    if (uniforms.texture_type == VX_DISPLAY_TEXTURE_SKY_VIEW)
    {
        float2 const sky_uv = float2(pixel_uv.x, 1.0 - pixel_uv.y);
        float3 const sky = sky_view_tex.SampleLevel(sky_view_sampler, sky_uv, 0.0).rgb;
        return float4(aces_filmic(sky), 1.0);
    }
    if (uniforms.texture_type == VX_DISPLAY_TEXTURE_PATH_TRACE ||
        uniforms.texture_type == VX_DISPLAY_TEXTURE_PATH_TRACE_SHADING)
    {
        float const depth = depth_tex.SampleLevel(depth_sampler, pixel_uv, 0.0).r;
        if (depth >= 1.0)
        {
            return BACKGROUND_COLOR;
        }

        int3 const texel = int3(pixel, 0);
        uint const spatial_index = spatial_index_tex.Load(texel).r;
        if (spatial_index == VX_SPATIAL_HASH_INVALID_INDEX)
        {
            return CACHE_FAILURE_COLOR;
        }

        uint4 const  payload = path_trace_payloads[spatial_index];
        float const  sample_count = max((float)payload.w, 1.0);
        float3 const shading =
            decode_fixed_point(payload.xyz, VX_PATH_TRACE_MAX_SAMPLE_SHADING * sample_count,
                               VX_PATH_TRACE_ACCUMULATION_SCALE) /
            sample_count;
        if (uniforms.texture_type == VX_DISPLAY_TEXTURE_PATH_TRACE_SHADING)
        {
            float3 const exposed_shading = uniforms.exposure * shading;
            return float4(exposed_shading / (1.0 + exposed_shading), 1.0);
        }

        float3 const albedo = unpack_albedo(albedo_tex.Load(texel).r).rgb;

        return float4(aces_filmic(uniforms.exposure * shading * albedo), 1.0);
    }
    float const surface_depth = depth_tex.SampleLevel(depth_sampler, pixel_uv, 0.0).r;
    if (uniforms.texture_type == VX_DISPLAY_TEXTURE_ALBEDO ||
        uniforms.texture_type == VX_DISPLAY_TEXTURE_AMBIENT_VISIBILITY ||
        uniforms.texture_type == VX_DISPLAY_TEXTURE_NORMAL ||
        uniforms.texture_type == VX_DISPLAY_TEXTURE_CELL_SIZE ||
        uniforms.texture_type == VX_DISPLAY_TEXTURE_SPATIAL_INDEX)
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
        if (uniforms.texture_type == VX_DISPLAY_TEXTURE_SPATIAL_INDEX)
        {
            uint const index = spatial_index_tex.Load(texel).r;
            return index == 0xFFFFFFFFu ? CACHE_FAILURE_COLOR
                                        : float4(spatial_index_color(index), 1.0);
        }

        float const visibility = visibility_tex.SampleLevel(visibility_sampler, pixel_uv, 0.0).r;
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

    return BACKGROUND_COLOR;
}
