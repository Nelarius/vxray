#include "path_tracer.h"

StructuredBuffer<uint> input_ray_count : register(t0, space0);

RWStructuredBuffer<uint>  output_ray_count : register(u0, space1);
RWStructuredBuffer<uint4> dispatch_threadgroups : register(u1, space1);

[numthreads(1, 1, 1)] void main(uint const thread_id : SV_DispatchThreadID) {
    if (thread_id != 0u)
    {
        return;
    }

    uint const group_count = (input_ray_count[0] + VX_WAVEFRONT_EXTEND_THREAD_COUNT - 1u) /
                             VX_WAVEFRONT_EXTEND_THREAD_COUNT;
    dispatch_threadgroups[0] = uint4(group_count, 1u, 1u, 0u);
    output_ray_count[0] = 0u;
}
