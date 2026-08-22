#include "constants.h"
#include "rtao.h"

#include "spatial_hash.hlsli"

struct ps_input
{
    float4 position : SV_Position;
};

Texture2D<float>         depth_tex : register(t0, space2);
Texture2D<uint>          normal_tex : register(t1, space2);
Texture3D<uint>          voxel_mask_tex : register(t2, space2);
RWStructuredBuffer<uint> hash_checksums : register(u3, space2);
RWStructuredBuffer<uint> hash_payloads : register(u4, space2);
RWStructuredBuffer<uint> hash_frames : register(u5, space2);

SamplerState depth_sampler : register(s0, space2);

ConstantBuffer<rtao_uniforms> uniforms : register(b0, space3);

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
    float const a1 = 0.7548777f;
    float const a2 = 0.5698403f;
    return frac(float2(n * a1, n * a2));
}

float3 sample_cosine_weighted_hemisphere(float2 const u)
{
    float const phi = 2.0f * VX_PI_F * u.x;
    float const sin_theta = sqrt(u.y);
    return float3(cos(phi) * sin_theta, sin(phi) * sin_theta, sqrt(1.f - u.y));
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

int mask_bit_index(int16_t3 const cell, int const ext)
{
    int x = cell.x & (ext - 1);
    int y = cell.y & (ext - 1);
    int z = cell.z & (ext - 1);
    return x + y * ext + z * ext * ext;
}

bool dda(float3 const pos, float3 const dir, float const t_max)
{
    float3 const inv_dir = 1.0 / dir;
    float3 const delta_dist = abs(inv_dir);

    float3 const grid_pos = floor(pos);
    float3 const ray_sign = sign(dir);
    float3 const next_pos = grid_pos + max(ray_sign, (float3)0.0);
    float3       side_dist = (next_pos - pos) * ray_sign * delta_dist;
    side_dist.x = ray_sign.x == 0.0 ? 3e+38 : side_dist.x;
    side_dist.y = ray_sign.y == 0.0 ? 3e+38 : side_dist.y;
    side_dist.z = ray_sign.z == 0.0 ? 3e+38 : side_dist.z;

    min16int3 const step_sign = min16int3(ray_sign);
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
        uint const mask = voxel_mask_tex.Load(int4((int3)voxel / VX_MASK_EXT, 0)).r;
        uint const idx = mask_bit_index(voxel, VX_MASK_EXT);
        uint const bit = 1u << (idx & 31u);
        if ((mask & bit) != 0)
        {
            return true;
        }
        float3 const axis_mask = step(side_dist, min(side_dist.yzx, side_dist.zxy));
        side_dist += axis_mask * delta_dist;
        voxel += min16int3(axis_mask) * step_sign;
    }
}

uint spatial_hash_find_or_insert(spatial_hash_key const key)
{
    uint const frame = uniforms.frame_index;
    uint       index = key.hash & VX_AO_HASH_MASK;
    for (uint probe = 0; probe < VX_AO_HASH_PROBE_COUNT; ++probe)
    {
        uint ex_checksum;
        InterlockedCompareExchange(hash_checksums[index], 0, key.checksum, ex_checksum);

        uint ex_frame;
        if (ex_checksum == 0 || ex_checksum == key.checksum)
        {
            InterlockedExchange(hash_frames[index], frame, ex_frame);
            return index;
        }
        ex_frame = hash_frames[index];
        if (frame - ex_frame > VX_AO_MAX_CELL_AGE)
        {
            uint ex;
            InterlockedExchange(hash_checksums[index], key.checksum, ex);
            InterlockedExchange(hash_payloads[index], 0, ex);
            InterlockedExchange(hash_frames[index], frame, ex_frame);
            return index;
        }

        index = (index + 1u) & VX_AO_HASH_MASK;
    }

    return 0xFFFFFFFFu;
}

float main(ps_input const input) : SV_Target0
{
    uint width, height;
    depth_tex.GetDimensions(width, height);
    uint2 const  pixel = min(uint2(input.position.xy), uint2(width - 1u, height - 1u));
    float2 const uv = (float2(pixel) + 0.5) / float2(width, height);
    float const  depth = depth_tex.SampleLevel(depth_sampler, uv, 0.0).r;
    if (depth >= 1.0)
    {
        return -1.0;
    }

    float3 const normal = unpack_normal(normal_tex.Load(int3(pixel, 0)).r);
    float3 const face_position =
        reconstruct_position(uniforms.inverse_view_projection, uv, depth, normal);
    float const view_depth = linear_view_depth(depth, uniforms.near_plane, uniforms.far_plane);
    float const cell_size = compute_cell_size(view_depth, uniforms.vertical_fov,
                                              uniforms.render_height, uniforms.sp, uniforms.smin);
    spatial_hash_key const key = make_spatial_hash_key(face_position, normal, cell_size);

    uint const index = spatial_hash_find_or_insert(key);
    if (index == 0xFFFFFFFFu)
    {
        return -1.0;
    }

    uint const payload = hash_payloads[index];
    uint const sample_count = payload & 0xFFFFu;
    if (sample_count >= VX_AO_SAMPLE_LIMIT)
    {
        return 1.0 - (float)(payload >> 16u) / (float)sample_count;
    }

    float3 const pos = offset_ray(face_position, normal);
    uint2 const  frame_seed =
        uint2(uniforms.frame_index * 0x9E3779B9u, pcg(uniforms.frame_index ^ 0xA511E9B3u));
    float2 const pixel_noise = as_normalized_float(pcg2d(pixel ^ frame_seed));
    uint         occlusion_count = 0u;
    for (uint i = 0u; i < VX_AO_RAYS_PER_PIXEL; ++i)
    {
        float2 const u = frac(pixel_noise + r2_sequence((float)i));
        float3 const direction =
            orient_sample_direction(sample_cosine_weighted_hemisphere(u), normal);
        if (dda(pos, direction, uniforms.rtao_radius))
        {
            ++occlusion_count;
        }
    }
    uint ex_occlusion;
    InterlockedAdd(hash_payloads[index], (occlusion_count << 16u) | VX_AO_RAYS_PER_PIXEL,
                   ex_occlusion);
    return 1.0 - (float)((ex_occlusion >> 16u) + occlusion_count) /
                     (float)((ex_occlusion & 0xFFFFu) + VX_AO_RAYS_PER_PIXEL);
}
