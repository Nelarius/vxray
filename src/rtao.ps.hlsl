#include "constants.h"
#include "rtao.h"

#include "shared.hlsli"

ConstantBuffer<rtao_uniforms> uniforms : register(b0, space3);

Texture2D<float> depth : register(t0, space2);
Texture2D<uint>  normals : register(t1, space2);
Texture3D<uint>  voxels : register(t2, space2);
SamplerState     depth_sampler : register(s0, space2);

uint2 pcg2d(uint2 v)
{
    // Hash functions for GPU rendering:
    // https://www.shadertoy.com/view/XlGcRh

    v = v * 1664525u + 1013904223u;

    v.x += v.y * 1664525u;
    v.y += v.x * 1664525u;

    v ^= v >> 16u;

    v.x += v.y * 1664525u;
    v.y += v.x * 1664525u;

    v ^= v >> 16u;

    return v;
}

// Converts an unsigned integer into a float in the range [0, 1) by using the 23 most significant
// bits as the mantissa.
float2 as_normalized_float(uint2 x) { return asfloat(0x3f800000u | (x >> 9u)) - 1.0f; }

float2 r2_sequence(float const n)
{
    // 2-dimensional golden ratio additive recurrence sequence
    // https://extremelearning.com.au/unreasonable-effectiveness-of-quasirandom-sequences/
    float const a1 = 0.7548777f;
    float const a2 = 0.5698403f;
    return frac(float2(n * a1, n * a2));
}

// `u` contains independent random numbers in [0, 1].
float3 sample_cosine_weighted_hemisphere(float2 const u)
{
    float const phi = 2.0f * VX_PI_F * u.x;
    float const sin_theta = sqrt(u.y);

    float const x = cos(phi) * sin_theta;
    float const y = sin(phi) * sin_theta;
    float const z = sqrt(1.f - u.y);

    return float3(x, y, z);
}

float3 orient_sample_direction(float3 const v, float3 const n)
{
    if (n.x != 0.0)
    {
        return float3(v.z * n.x, v.x, v.y);
    }

    if (n.y != 0.0)
    {
        return float3(v.x, v.z * n.y, v.y);
    }

    return float3(v.x, v.y, v.z * n.z);
}

bool dda(float3 const pos, float3 const dir, float const t_max)
{
    // Adapted from https://www.shadertoy.com/view/lfyGRW

    float3 const    grid_pos = floor(pos);
    float3 const    delta_dist = 1.0 / max(abs(dir), (float3)1e-20);
    float3 const    ray_sign = sign(dir);
    min16int3 const step_sign = min16int3(ray_sign);
    float3 const    next_pos = grid_pos + max(ray_sign, (float3)0.0);
    float3 const    zero_dir_guard = (1.0 - abs(ray_sign)) * 3e+38; // 3e+38 if ray_sign.x == 0
    float3          side_dist = (next_pos - pos) * ray_sign * delta_dist + zero_dir_guard;
    min16int3       voxel = min16int3(grid_pos);
    for (;;)
    {
        if (any((min16uint3)voxel >= (min16uint)uniforms.grid_ext))
        {
            return false;
        }

        float const t = min(side_dist.x, min(side_dist.y, side_dist.z));
        if (t > t_max)
        {
            return false;
        }

        if (voxels.Load(int4(voxel, 0)).r != 0)
        {
            return true;
        }

        float3 const axis_mask = step(side_dist, min(side_dist.yzx, side_dist.zxy));
        side_dist += axis_mask * delta_dist;
        voxel += min16int3(axis_mask) * step_sign;
    }
}

struct ps_input
{
    float2 uv : TEXCOORD0;
};

struct ps_output
{
    float visibility : SV_Target0;
};

ps_output main(ps_input const input)
{
    uint width, height;
    depth.GetDimensions(width, height);
    uint2 const pixel = min(uint2(input.uv * uint2(width, height)), uint2(width - 1u, height - 1u));
    float const surface_depth = depth.SampleLevel(depth_sampler, input.uv, 0.0).r;
    if (surface_depth >= 1.0)
    {
        ps_output background;
        background.visibility = 1.0;
        return background;
    }

    float3 const normal = unpack_normal(normals.Load(int3(pixel, 0)).r);

    // NOTE: snaps the reconstructed position to the voxel face to work around precision issues from
    // unprojected depth
    float2 const ndc = input.uv * float2(2.0, -2.0) + float2(-1.0, 1.0);
    float3 const reconstructed_pos =
        unproject(uniforms.inverse_view_projection, ndc, surface_depth);
    float3 const face_axis = abs(normal);
    float3 const face_pos = lerp(reconstructed_pos, round(reconstructed_pos), face_axis);
    float3 const pos = offset_ray(face_pos, normal);
    float2 const pxl_noise = as_normalized_float(pcg2d(pixel));

    int occlusion_count = 0;
    for (int i = 0; i < 4; ++i)
    {
        float2 const u = frac(pxl_noise + r2_sequence((float)i));
        float3 const v = sample_cosine_weighted_hemisphere(u);
        float3 const dir = orient_sample_direction(v, normal);
        if (dda(pos, dir, uniforms.rtao_radius))
        {
            ++occlusion_count;
        }
    }

    ps_output output;
    output.visibility = 1.0 - (float)occlusion_count / 4.0;
    return output;
}
