#include "rtao.h"

#include "spatial_hash.hlsli"

struct ps_input
{
    float4 position : SV_Position;
};

Texture2D<float>       depth_tex : register(t0, space2);
Texture2D<uint>        normal_tex : register(t1, space2);
StructuredBuffer<uint> hash_checksums : register(t2, space2);
StructuredBuffer<uint> hash_payloads : register(t3, space2);

ConstantBuffer<rtao_uniforms> uniforms : register(b0, space3);

static float const VX_AO_CACHE_ERROR = -1.0;

float main(ps_input const input) : SV_Target0
{
    uint width, height;
    depth_tex.GetDimensions(width, height);
    uint2 const  pixel = min(uint2(input.position.xy), uint2(width - 1u, height - 1u));
    float2 const uv = (float2(pixel) + 0.5) / float2(width, height);
    float const  depth = depth_tex.Load(int3(pixel, 0)).r;
    if (depth >= 1.0)
    {
        return VX_AO_CACHE_ERROR;
    }

    float3 const normal = unpack_normal(normal_tex.Load(int3(pixel, 0)).r);
    float3 const face_position =
        reconstruct_position(uniforms.inverse_view_projection, uv, depth, normal);
    float const view_depth = linear_view_depth(depth, uniforms.near_plane, uniforms.far_plane);
    float const cell_size = compute_cell_size(view_depth, uniforms.vertical_fov,
                                              uniforms.render_height, uniforms.sp, uniforms.smin);
    spatial_hash_key const key = make_spatial_hash_key(face_position, normal, cell_size);

    uint index = key.hash & VX_AO_HASH_MASK;
    for (uint probe = 0u; probe < VX_AO_HASH_PROBE_COUNT; ++probe)
    {
        uint const checksum = hash_checksums[index];
        if (checksum == key.checksum)
        {
            uint const payload = hash_payloads[index];
            uint const sample_count = payload & 0xffffu;
            if (sample_count == 0u)
            {
                return VX_AO_CACHE_ERROR;
            }
            return 1.0 - (float)(payload >> 16u) / (float)sample_count;
        }
        if (checksum == 0u)
        {
            return VX_AO_CACHE_ERROR;
        }
        index = (index + 1u) & VX_AO_HASH_MASK;
    }
    return VX_AO_CACHE_ERROR;
}
