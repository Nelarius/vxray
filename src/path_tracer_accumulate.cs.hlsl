#include "path_tracer.h"

#include "spatial_hash.hlsli"

Texture2D<uint>                          albedo_tex : register(t0, space0);
StructuredBuffer<uint>                   input_path_indices : register(t1, space0);
StructuredBuffer<path_tracer_path_state> path_state_buffer : register(t2, space0);
StructuredBuffer<uint>                   input_path_count : register(t3, space0);

RWStructuredBuffer<uint4> hash_payloads : register(u0, space1);

[numthreads(VX_WAVEFRONT_EXTEND_THREAD_COUNT, 1, 1)] void
main(uint const thread_id : SV_DispatchThreadID) {
    if (thread_id >= input_path_count[0])
    {
        return;
    }

    uint const                   path_index = input_path_indices[thread_id];
    path_tracer_path_state const state = path_state_buffer[path_index];
    uint const                   spatial_index = asuint(state.throughput_and_spatial_index.w);

    uint width, height;
    albedo_tex.GetDimensions(width, height);
    uint2 const  pixel = uint2(path_index % width, path_index / width);
    float3 const albedo = unpack_albedo(albedo_tex.Load(int3(pixel, 0)).r).rgb;
    float3 const shading = state.radiance.xyz / albedo;
    uint3 const  encoded = encode_fixed_point(shading, VX_PATH_TRACE_MAX_SAMPLE_SHADING,
                                              VX_PATH_TRACE_ACCUMULATION_SCALE);
    uint         ex;
    InterlockedAdd(hash_payloads[spatial_index].x, encoded.x, ex);
    InterlockedAdd(hash_payloads[spatial_index].y, encoded.y, ex);
    InterlockedAdd(hash_payloads[spatial_index].z, encoded.z, ex);
    InterlockedAdd(hash_payloads[spatial_index].w, 1u, ex);
}
