#include "constants.h"
#include "path_tracer.h"

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

Texture2D<float>          depth_tex : register(t0, space2);
Texture2D<uint>           normal_tex : register(t1, space2);
Texture2D<uint>           previous_index_tex : register(t2, space2);
Texture2D<uint>           previous_checksum_tex : register(t3, space2);
RWStructuredBuffer<uint>  hash_checksums : register(u4, space2);
RWStructuredBuffer<uint4> hash_payloads : register(u5, space2);
RWStructuredBuffer<uint>  hash_frames : register(u6, space2);

SamplerState depth_sampler : register(s0, space2);

ConstantBuffer<path_tracer_index_uniforms> uniforms : register(b0, space3);

ps_output miss()
{
    ps_output output;
    output.index = VX_SPATIAL_HASH_INVALID_INDEX;
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
    uint                   index = spatial_hash_find_reprojected_index(
        face_position, key, uint2(width, height), uniforms.history_valid,
        uniforms.previous_view_projection, uniforms.frame, VX_PATH_TRACE_SPATIAL_HASH_TOUCH_PERIOD,
        previous_index_tex, previous_checksum_tex, hash_frames);
    if (index == VX_SPATIAL_HASH_INVALID_INDEX)
    {
        index = spatial_hash_find_or_insert(key, uniforms.frame, VX_PATH_TRACE_SPATIAL_HASH_MASK,
                                            VX_PATH_TRACE_SPATIAL_HASH_PROBE_COUNT,
                                            VX_PATH_TRACE_SPATIAL_HASH_MAX_CELL_AGE, hash_checksums,
                                            hash_payloads, hash_frames);
    }

    ps_output output;
    output.index = index;
    output.checksum = index == VX_SPATIAL_HASH_INVALID_INDEX ? 0u : key.checksum;
    return output;
}
