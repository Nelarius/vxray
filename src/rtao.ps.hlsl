#include "constants.h"
#include "rtao.h"

#include "spatial_hash.hlsli"

struct ps_input
{
    float4 position : SV_Position;
};

Texture2D<float>         depth_tex : register(t0, space2);
Texture2D<uint>          normal_tex : register(t1, space2);
Texture3D<uint2>         voxel_masks : register(t2, space2);
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

// NOTE: ext is expected to be a power of two
int mask_linear_idx(int16_t3 const coord, int const ext)
{
    int const x = coord.x & (ext - 1);
    int const y = coord.y & (ext - 1);
    int const z = coord.z & (ext - 1);
    return x + y * ext + z * ext * ext;
}

bool mask_bit_test(uint2 const mask, int const index)
{
    uint const word = mask[index >> 5u];
    uint const bit = 1u << (index & 31u);
    return (word & bit) != 0;
}

bool sparse_ray_march(float3 const ray_origin, float3 const ray_dir, float const t_max)
{
    float3 const inv_dir = 1.0 / ray_dir;
    int16_t3     ipos = int16_t3(floor(ray_origin));
    float3       local_pos = ray_origin - float3(ipos);
    float        distance = 0.0;

    for (int i = 0; i < 3 * uniforms.grid_ext; ++i)
    {
        if (any((uint16_t3)ipos >= (uint16_t)uniforms.grid_ext))
        {
            return false;
        }

        int const   mask_idx = mask_linear_idx(ipos, 4);
        uint2 const mask = voxel_masks.Load(int4(ipos >> 2, 0)).rg;
        if (mask_bit_test(mask, mask_idx))
        {
            return true;
        }

        int16_t lod;
        if ((mask.x | mask.y) == 0u)
        {
            lod = 4;
        }
        else
        {
            uint const mask_part = mask_idx < 32 ? mask.x : mask.y;
            // 0x0A preserves the high bits of the 2x2x2 block's local x and y coordinates.
            // 0x00330033 selects the relevant 2x2x2 bits.
            lod = ((mask_part >> (mask_idx & 0x0A)) & 0x00330033u) == 0u ? 2 : 1;
        }

        int16_t const  cell_mask = lod - 1;
        int16_t3 const cell_min = ipos & ~cell_mask;
        int16_t3 const cell_max = cell_min + lod;
        float3 const   exit_plane = float3(ray_dir.x < 0.0 ? cell_min.x : cell_max.x,
                                           ray_dir.y < 0.0 ? cell_min.y : cell_max.y,
                                           ray_dir.z < 0.0 ? cell_min.z : cell_max.z);

        float3 side_dist = (exit_plane - float3(ipos) - local_pos) * inv_dir;
        side_dist.x = ray_dir.x == 0.0 ? 3e+38 : side_dist.x;
        side_dist.y = ray_dir.y == 0.0 ? 3e+38 : side_dist.y;
        side_dist.z = ray_dir.z == 0.0 ? 3e+38 : side_dist.z;
        float const t = min(side_dist.x, min(side_dist.y, side_dist.z));
        distance += t;
        if (distance > t_max)
        {
            return false;
        }

        float3 const   crossed = step(side_dist, t);
        float3 const   advanced = local_pos + t * ray_dir;
        int16_t3 const cell_delta = int16_t3(floor(advanced + crossed * sign(ray_dir) * 0.5));
        ipos += cell_delta;
        local_pos = advanced - float3(cell_delta);
    }

    return false;
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
        if (sparse_ray_march(pos, direction, uniforms.rtao_radius))
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
