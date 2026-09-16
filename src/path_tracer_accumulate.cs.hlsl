#include "path_tracer.h"

#include "sharc.hlsli"

RWStructuredBuffer<uint>  hash_checksums : register(u0, space1);
RWStructuredBuffer<uint4> sharc_accumulation : register(u1, space1);
RWStructuredBuffer<uint4> sharc_resolved : register(u2, space1);
RWStructuredBuffer<uint>  hash_frames : register(u3, space1);

ConstantBuffer<path_tracer_uniforms> uniforms : register(b0, space2);

[numthreads(VX_WAVEFRONT_EXTEND_THREAD_COUNT, 1, 1)] void
main(uint const index : SV_DispatchThreadID) {
    if (index >= VX_PATH_TRACE_SPATIAL_HASH_CAPACITY)
    {
        return;
    }

    uint4 const accumulated = sharc_accumulation[index];
    if (accumulated.w > 0u)
    {
        uint4 const  previous = sharc_resolved[index];
        float const  history_count = min((float)previous.w, (float)VX_SHARC_HISTORY_SAMPLE_COUNT);
        float const  sample_count = (float)accumulated.w;
        float3 const history = sharc_resolved_shading(previous);
        float3 const sample =
            decode_fixed_point(accumulated.xyz, VX_PATH_TRACE_MAX_SAMPLE_SHADING * sample_count,
                               VX_PATH_TRACE_ACCUMULATION_SCALE) /
            sample_count;
        float const  total_count = history_count + sample_count;
        float3 const resolved =
            (history * history_count + sample * sample_count) / max(total_count, 1.0);
        uint3 const encoded = encode_fixed_point(resolved, VX_PATH_TRACE_MAX_SAMPLE_SHADING,
                                                 VX_PATH_TRACE_ACCUMULATION_SCALE);
        sharc_resolved[index] =
            uint4(encoded, min((uint)total_count, VX_SHARC_HISTORY_SAMPLE_COUNT));
    }
    else if (hash_checksums[index] != 0u &&
             uniforms.frame - hash_frames[index] > VX_PATH_TRACE_SPATIAL_HASH_MAX_CELL_AGE)
    {
        sharc_resolved[index] = (uint4)0u;
        hash_frames[index] = 0u;
        hash_checksums[index] = 0u;
    }

    sharc_accumulation[index] = (uint4)0u;
}
