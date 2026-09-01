#include "constants.h"
#include "rtao.h"

#include "spatial_hash.hlsli"

struct ps_input
{
    float4 position : SV_Position;
};

struct ps_output
{
    uint index : SV_Target0;
    uint checksum : SV_Target1;
};

Texture2D<float>         depth_tex : register(t0, space2);
Texture2D<uint>          normal_tex : register(t1, space2);
Texture2D<uint>          previous_index_tex : register(t2, space2);
Texture2D<uint>          previous_checksum_tex : register(t3, space2);
RWStructuredBuffer<uint> hash_checksums : register(u4, space2);
RWStructuredBuffer<uint> hash_payloads : register(u5, space2);
RWStructuredBuffer<uint> hash_frames : register(u6, space2);

SamplerState depth_sampler : register(s0, space2);

ConstantBuffer<rtao_uniforms> uniforms : register(b0, space3);

static uint const INVALID_INDEX = 0xFFFFFFFFu;

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

    return INVALID_INDEX;
}

void touch_reprojected_entry(uint const index, uint const hash)
{
    uint const touch_mask = VX_AO_HASH_TOUCH_PERIOD - 1u;
    // NOTE: scramble frame update phase using hash to spread updates
    if ((uniforms.frame_index & touch_mask) == (hash & touch_mask))
    {
        uint ex_frame;
        InterlockedExchange(hash_frames[index], uniforms.frame_index, ex_frame);
    }
}

uint find_reprojected_index(float3 const face_position, spatial_hash_key const key,
                            uint2 const dimensions)
{
    if (uniforms.history_valid == 0u)
    {
        return INVALID_INDEX;
    }

    float4 const previous_clip = mul(uniforms.previous_view_projection, float4(face_position, 1.0));
    if (previous_clip.w <= 0.0)
    {
        return INVALID_INDEX;
    }

    float2 const previous_ndc = previous_clip.xy / previous_clip.w;
    float2 const previous_uv = previous_ndc * float2(0.5, -0.5) + 0.5;
    if (any(previous_uv < 0.0) || any(previous_uv >= 1.0))
    {
        return INVALID_INDEX;
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
            if (index != INVALID_INDEX)
            {
                touch_reprojected_entry(index, key.hash);
                return index;
            }
        }
    }

    return INVALID_INDEX;
}

ps_output miss()
{
    ps_output output;
    output.index = INVALID_INDEX;
    output.checksum = 0u;
    return output;
}

ps_output main(ps_input const input)
{
    uint width, height;
    depth_tex.GetDimensions(width, height);
    uint2 const  pixel = min(uint2(input.position.xy), uint2(width - 1u, height - 1u));
    float2 const uv = (float2(pixel) + 0.5) / float2(width, height);
    float const  depth = depth_tex.SampleLevel(depth_sampler, uv, 0.0).r;
    if (depth >= 1.0)
    {
        return miss();
    }

    float3 const normal = unpack_normal(normal_tex.Load(int3(pixel, 0)).r);
    float3 const face_position =
        reconstruct_position(uniforms.inverse_view_projection, uv, depth, normal);
    float const view_depth = linear_view_depth(depth, uniforms.near_plane, uniforms.far_plane);
    float const cell_size = compute_cell_size(view_depth, uniforms.vertical_fov,
                                              uniforms.render_height, uniforms.sp, uniforms.smin);
    spatial_hash_key const key = make_spatial_hash_key(face_position, normal, cell_size);
    uint                   index = find_reprojected_index(face_position, key, uint2(width, height));
    if (index == INVALID_INDEX)
    {
        index = spatial_hash_find_or_insert(key);
    }

    ps_output output;
    output.index = index;
    output.checksum = index == INVALID_INDEX ? 0u : key.checksum;
    return output;
}
