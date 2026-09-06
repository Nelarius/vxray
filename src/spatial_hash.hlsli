#pragma once

#include "shared.hlsli"

float linear_view_depth(float const device_depth, float const near_plane, float const far_plane)
{
    return near_plane * far_plane / (far_plane - device_depth * (far_plane - near_plane));
}

float compute_cell_size(float const view_depth, float const vertical_fov, uint const render_height,
                        float const sp, float const smin)
{
    float const h = view_depth * tan(vertical_fov * 0.5);
    float const sw = sp * (h * 2.0) / (float)render_height;
    float const exponent = floor(log2(sw / smin));
    return pow(2.0, exponent) * smin;
}

float3 reconstruct_position(float4x4 const inverse_view_projection, float2 const uv,
                            float const depth, float3 const normal)
{
    float2 const ndc = uv * float2(2.0, -2.0) + float2(-1.0, 1.0);
    float3 const reconstructed_pos = unproject(inverse_view_projection, ndc, depth);
    float3 const face_axis = abs(normal);
    return lerp(reconstructed_pos, round(reconstructed_pos), face_axis);
}

struct spatial_hash_key
{
    uint hash;
    uint checksum;
};

spatial_hash_key make_spatial_hash_key(float3 const position, float3 const normal,
                                       float const cell_size)
{
    // Adapted from
    // https://interplayoflight.wordpress.com/2025/11/23/spatial-hashing-for-raytraced-ambient-occlusion/

    int3 const p = int3(floor(position / cell_size));
    int3 const n = int3(3.0 * normal);
    uint const size = asuint(1000.0 * cell_size);

    uint const hash = pcg(size + pcg(p.x + pcg(p.y + pcg(p.z + pcg(n.x + pcg(n.y + pcg(n.z)))))));
    uint       checksum = xxhash32(
        size +
        xxhash32(p.x +
                 xxhash32(p.y + xxhash32(p.z + xxhash32(n.x + xxhash32(n.y + xxhash32(n.z)))))));
    checksum = max(checksum, 1);

    spatial_hash_key result;
    result.hash = hash;
    result.checksum = checksum;
    return result;
}

static uint const VX_SPATIAL_HASH_INVALID_INDEX = 0xFFFFFFFFu;

void spatial_hash_reset_payload(RWStructuredBuffer<uint> hash_payloads, uint const index)
{
    uint ex;
    InterlockedExchange(hash_payloads[index], 0u, ex);
}

void spatial_hash_reset_payload(RWStructuredBuffer<uint4> hash_payloads, uint const index)
{
    uint ex;
    InterlockedExchange(hash_payloads[index].x, 0u, ex);
    InterlockedExchange(hash_payloads[index].y, 0u, ex);
    InterlockedExchange(hash_payloads[index].z, 0u, ex);
    InterlockedExchange(hash_payloads[index].w, 0u, ex);
}

template <typename Payload>
uint spatial_hash_find_or_insert(spatial_hash_key const key, uint const frame, uint const mask,
                                 uint const probe_count, uint const max_cell_age,
                                 RWStructuredBuffer<uint>    hash_checksums,
                                 RWStructuredBuffer<Payload> hash_payloads,
                                 RWStructuredBuffer<uint>    hash_frames)
{
    uint index = key.hash & mask;
    for (uint probe = 0u; probe < probe_count; ++probe)
    {
        uint ex_checksum;
        InterlockedCompareExchange(hash_checksums[index], 0u, key.checksum, ex_checksum);

        uint ex_frame;
        if (ex_checksum == 0u || ex_checksum == key.checksum)
        {
            InterlockedExchange(hash_frames[index], frame, ex_frame);
            return index;
        }
        ex_frame = hash_frames[index];
        if (frame - ex_frame > max_cell_age)
        {
            uint ex;
            InterlockedExchange(hash_checksums[index], key.checksum, ex);
            spatial_hash_reset_payload(hash_payloads, index);
            InterlockedExchange(hash_frames[index], frame, ex_frame);
            return index;
        }

        index = (index + 1u) & mask;
    }

    return VX_SPATIAL_HASH_INVALID_INDEX;
}

void spatial_hash_touch_reprojected_entry(uint const index, uint const hash, uint const frame,
                                          uint const               touch_period,
                                          RWStructuredBuffer<uint> hash_frames)
{
    uint const touch_mask = touch_period - 1u;
    // NOTE: scramble frame update phase using hash to spread updates
    if ((frame & touch_mask) == (hash & touch_mask))
    {
        uint ex_frame;
        InterlockedExchange(hash_frames[index], frame, ex_frame);
    }
}

uint spatial_hash_find_reprojected_index(float3 const face_position, spatial_hash_key const key,
                                         uint2 const dimensions, uint const history_valid,
                                         float4x4 const previous_view_projection, uint const frame,
                                         uint const               touch_period,
                                         Texture2D<uint>          previous_index_tex,
                                         Texture2D<uint>          previous_checksum_tex,
                                         RWStructuredBuffer<uint> hash_frames)
{
    if (history_valid == 0u)
    {
        return VX_SPATIAL_HASH_INVALID_INDEX;
    }

    float4 const previous_clip = mul(previous_view_projection, float4(face_position, 1.0));
    if (previous_clip.w <= 0.0)
    {
        return VX_SPATIAL_HASH_INVALID_INDEX;
    }

    float2 const previous_ndc = previous_clip.xy / previous_clip.w;
    float2 const previous_uv = previous_ndc * float2(0.5, -0.5) + 0.5;
    if (any(previous_uv < 0.0) || any(previous_uv >= 1.0))
    {
        return VX_SPATIAL_HASH_INVALID_INDEX;
    }

    int2 const base_pixel = int2(floor(previous_uv * float2(dimensions) - 0.5));
    for (int y = 0; y < 2; ++y)
    {
        for (int x = 0; x < 2; ++x)
        {
            int2 const previous_pixel = base_pixel + int2(x, y);
            if (any(previous_pixel < 0) || any(previous_pixel >= int2(dimensions)))
            {
                continue;
            }

            int3 const texel = int3(previous_pixel, 0);
            if (previous_checksum_tex.Load(texel).r != key.checksum)
            {
                continue;
            }

            uint const index = previous_index_tex.Load(texel).r;
            if (index != VX_SPATIAL_HASH_INVALID_INDEX)
            {
                spatial_hash_touch_reprojected_entry(index, key.hash, frame, touch_period,
                                                     hash_frames);
                return index;
            }
        }
    }

    return VX_SPATIAL_HASH_INVALID_INDEX;
}
