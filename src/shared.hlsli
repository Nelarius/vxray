#pragma once

#include "constants.h"

uint pcg(uint v)
{
    uint const state = v * 747796405u + 2891336453u;
    uint const word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

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

float4 unpack_albedo(uint const rgba)
{
    float4 const srgb =
        float4(rgba & 255u, (rgba >> 8u) & 255u, (rgba >> 16u) & 255u, (rgba >> 24u) & 255u) /
        255.0;
    return float4(pow(srgb.rgb, 2.2), srgb.a);
}

uint3 encode_fixed_point(float3 const value, float const max_value, float const scale)
{
    return uint3(clamp(value, 0.0, max_value) * scale + 0.5);
}

float3 decode_fixed_point(uint3 const value, float const max_value, float const scale)
{
    return min((float3)value / scale, (float3)max_value);
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

// Converts an unsigned integer into a float in [0, 1) using the 23 most significant bits as the
// mantissa.
float as_normalized_float(uint const x) { return asfloat(0x3F800000u | (x >> 9u)) - 1.0; }

template <int N> vector<float, N> as_normalized_float(vector<uint, N> const x)
{
    return asfloat(0x3F800000u | (x >> 9u)) - 1.0;
}

uint halton_prime(uint const dimension)
{
    switch (dimension)
    {
    case 0u:
        return 2u;
    case 1u:
        return 3u;
    case 2u:
        return 5u;
    case 3u:
        return 7u;
    case 4u:
        return 11u;
    case 5u:
        return 13u;
    case 6u:
        return 17u;
    case 7u:
        return 19u;
    case 8u:
        return 23u;
    case 9u:
        return 29u;
    case 10u:
        return 31u;
    case 11u:
        return 37u;
    case 12u:
        return 41u;
    case 13u:
        return 43u;
    case 14u:
        return 47u;
    case 15u:
        return 53u;
    case 16u:
        return 59u;
    case 17u:
        return 61u;
    case 18u:
        return 67u;
    case 19u:
        return 71u;
    case 20u:
        return 73u;
    case 21u:
        return 79u;
    case 22u:
        return 83u;
    case 23u:
        return 89u;
    case 24u:
        return 97u;
    case 25u:
        return 101u;
    default:
        return 103u;
    }
}

float radical_inverse(uint index, uint const base)
{
    float const inv_base = 1.0 / (float)base;
    float       inv_digit = inv_base;
    float       value = 0.0;
    while (index != 0u)
    {
        uint const digit = index % base;
        value += (float)digit * inv_digit;
        index /= base;
        inv_digit *= inv_base;
    }
    return min(value, 0.99999994);
}

float rotated_halton(uint const sample_index, uint const dimension, uint const stable_stream_id)
{
    if (dimension >= 27u)
    {
        // Fallback to white noise
        uint const hash = pcg(sample_index ^ pcg(dimension ^ pcg(stable_stream_id)));
        return as_normalized_float(hash);
    }

    // Adds a fixed rotation per stream and dimension. It rotates a halton sequence around the unit
    // interval.
    uint const  rotation_hash = pcg(stable_stream_id ^ pcg(dimension + 0x9E3779B9u));
    float const rotation = as_normalized_float(rotation_hash);
    float const h = radical_inverse(sample_index + 1u, halton_prime(dimension));
    return frac(h + rotation);
}

float2 halton_sample_2d(uint const frame, uint const bounce, uint const stable_stream_id)
{
    // Wrap around to prevent robustness issues with very large frames.
    uint const frame_idx = frame & 8191u;
    uint const first_dimension = 2u * bounce;
    return float2(rotated_halton(frame_idx, first_dimension, stable_stream_id),
                  rotated_halton(frame_idx, first_dimension + 1u, stable_stream_id));
}

float3 halton_sample_3d(uint const frame, uint const bounce, uint const stable_stream_id)
{
    // Wrap around to prevent robustness issues with very large frames.
    uint const frame_idx = frame & 8191u;
    uint const first_dimension = 3u * bounce;
    return float3(rotated_halton(frame_idx, first_dimension, stable_stream_id),
                  rotated_halton(frame_idx, first_dimension + 1u, stable_stream_id),
                  rotated_halton(frame_idx, first_dimension + 2u, stable_stream_id));
}

float3 sample_cone(float2 const u, float const cos_theta_max)
{
    float const cos_theta = 1.0 - u.x * (1.0 - cos_theta_max);
    float const sin_theta = sqrt(1.0 - cos_theta * cos_theta);
    float const phi = 2.0 * VX_PI_F * u.y;
    return float3(cos(phi) * sin_theta, sin(phi) * sin_theta, cos_theta);
}

float pdf_cone(float const cosine, float const cos_theta_max)
{
    return cosine >= cos_theta_max ? 1.0 / (2.0 * VX_PI_F * (1.0 - cos_theta_max)) : 0.0;
}

float3 sample_cosine_weighted_hemisphere(float2 const u)
{
    float const phi = 2.0 * VX_PI_F * u.x;
    float const sin_theta = sqrt(u.y);
    return float3(cos(phi) * sin_theta, sin(phi) * sin_theta, sqrt(1.0 - u.y));
}

float pdf_cosine_weighted_hemisphere(float const cosine) { return max(cosine, 0.0) / VX_PI_F; }

float3 orient_sample_direction(float3 const v, float3 const n)
{
    float3 const tangent = normalize(abs(n.z) < 0.999 ? cross(float3(0.0, 0.0, 1.0), n)
                                                      : cross(float3(0.0, 1.0, 0.0), n));
    float3 const bitangent = cross(n, tangent);
    return tangent * v.x + bitangent * v.y + n * v.z;
}

float3 orient_axis_aligned_sample_direction(float3 const v, float3 const n)
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
