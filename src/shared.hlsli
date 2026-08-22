#pragma once

int mask_bit_index(int16_t3 const cell, int const ext)
{
    int x = cell.x & (ext - 1);
    int y = cell.y & (ext - 1);
    int z = cell.z & (ext - 1);
    return x + y * ext + z * ext * ext;
}

uint pack_normal(float3 const normal)
{
    if (normal.x != 0.0)
    {
        return 1u | ((uint)(normal.x < 0.0) << 1u);
    }
    if (normal.y != 0.0)
    {
        return 4u | ((uint)(normal.y < 0.0) << 3u);
    }
    if (normal.z != 0.0)
    {
        return 16u | ((uint)(normal.z < 0.0) << 5u);
    }
    return 0u;
}

float3 unpack_normal(uint const packed)
{
    float3 const magnitude = float3(packed & 1u, (packed >> 2u) & 1u, (packed >> 4u) & 1u);
    float3 const negative = float3((packed >> 1u) & 1u, (packed >> 3u) & 1u, (packed >> 5u) & 1u);
    return magnitude * (1.0 - 2.0 * negative);
}

float3 unproject(float4x4 const inverse_view_projection, float2 const ndc, float const depth)
{
    float4 const world = mul(inverse_view_projection, float4(ndc, depth, 1.0));
    return world.xyz / world.w;
}

float3 offset_ray(float3 const p, float3 const n)
{
    // "A Fast and Robust Method for Avoiding Self-Intersection", Ray Tracing Gems
    float const int_scale = 256.0;
    float const float_scale = 1e-5;
    float const origin = 1e-3;

    int3 const   offset = int3(int_scale * n);
    float3 const po = float3(asfloat(asint(p.x) + ((p.x < 0.0) ? -offset.x : offset.x)),
                             asfloat(asint(p.y) + ((p.y < 0.0) ? -offset.y : offset.y)),
                             asfloat(asint(p.z) + ((p.z < 0.0) ? -offset.z : offset.z)));

    return float3((abs(p.x) < origin) ? p.x + float_scale * n.x : po.x,
                  (abs(p.y) < origin) ? p.y + float_scale * n.y : po.y,
                  (abs(p.z) < origin) ? p.z + float_scale * n.z : po.z);
}
