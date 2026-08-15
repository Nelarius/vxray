#pragma once

#include "rtao.h"

#include "shared.hlsli"

uint pcg(uint v)
{
    uint const state = v * 747796405u + 2891336453u;
    uint const word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

uint xxhash32(uint p)
{
    uint const prime32_2 = 2246822519u;
    uint const prime32_3 = 3266489917u;
    uint const prime32_4 = 668265263u;
    uint const prime32_5 = 374761393u;
    uint       h32 = p + prime32_5;
    h32 = prime32_4 * ((h32 << 17u) | (h32 >> 15u));
    h32 = prime32_2 * (h32 ^ (h32 >> 15u));
    h32 = prime32_3 * (h32 ^ (h32 >> 13u));
    return h32 ^ (h32 >> 16u);
}

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
