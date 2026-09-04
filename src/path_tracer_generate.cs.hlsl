#include "path_tracer.h"
#include "sky.h"

#include "spatial_hash.hlsli"

Texture2D<float> depth_tex : register(t0, space0);
Texture2D<uint>  spatial_index_tex : register(t1, space0);
Texture2D<uint>  albedo_tex : register(t2, space0);
Texture2D<uint>  normal_tex : register(t3, space0);
SamplerState     depth_sampler : register(s0, space0);

RWStructuredBuffer<path_tracer_ray>        output_ray_buffer : register(u0, space1);
RWStructuredBuffer<path_tracer_path_state> path_state_buffer : register(u1, space1);
RWStructuredBuffer<uint>                   output_ray_count : register(u2, space1);
RWStructuredBuffer<uint>                   output_path_indices : register(u3, space1);
RWStructuredBuffer<uint>                   output_path_count : register(u4, space1);
RWStructuredBuffer<uint4>                  hash_payloads : register(u5, space1);

ConstantBuffer<path_tracer_uniforms> uniforms : register(b0, space2);

[numthreads(VX_WAVEFRONT_SCREEN_THREAD_COUNT, VX_WAVEFRONT_SCREEN_THREAD_COUNT, 1)] void
main(uint2 const tid : SV_DispatchThreadID) {
    uint width, height;
    depth_tex.GetDimensions(width, height);
    if (tid.x >= width || tid.y >= height)
    {
        return;
    }

    uint const path_index = tid.y * width + tid.x;
    uint const spatial_index = spatial_index_tex.Load(int3(tid, 0)).r;

    float2 const uv = (float2(tid) + 0.5) / float2(width, height);
    float const  depth = depth_tex.SampleLevel(depth_sampler, uv, 0.0).r;
    if (depth >= 1.0 || spatial_index == VX_SPATIAL_HASH_INVALID_INDEX)
    {
        return;
    }

    uint4 const payload = hash_payloads[spatial_index];
    // NOTE: technically this is a race condition as we increment the payload at the end of the
    // pipeline. Multiple pixels may contribute to a single cell.
    if (payload.w >= VX_PATH_TRACE_SAMPLE_LIMIT)
    {
        return;
    }

    uint output_path_index;
    InterlockedAdd(output_path_count[0], 1u, output_path_index);
    output_path_indices[output_path_index] = path_index;

    float3 const normal = unpack_normal(normal_tex.Load(int3(tid, 0)).r);
    float3 const position =
        reconstruct_position(uniforms.inverse_view_projection, uv, depth, normal);
    float3 const albedo = unpack_albedo(albedo_tex.Load(int3(tid, 0)).r).rgb;
    float3       throughput = (float3)1.0;

    // Generate next direction (MIS, albertian and sun disk)

    float3 const samples = halton_sample_3d(uniforms.frame, 0u, path_index);
    float2 const u = samples.xy;
    bool const   sample_sun = samples.z < 0.5;
    float const  cos_theta_max = cos(VX_SKY_SOLAR_RADIUS_RAD);
    float3 const local_dir =
        sample_sun ? sample_cone(u, cos_theta_max) : sample_cosine_weighted_hemisphere(u);
    float3 const ray_dir = sample_sun
                               ? orient_sample_direction(local_dir, uniforms.sun_direction.xyz)
                               : orient_axis_aligned_sample_direction(local_dir, normal);
    float const  n_dot_l = max(dot(normal, ray_dir), 0.0);
    if (n_dot_l != 0.0)
    {
        float const pdf = 0.5 * (pdf_cone(dot(uniforms.sun_direction.xyz, ray_dir), cos_theta_max) +
                                 pdf_cosine_weighted_hemisphere(n_dot_l));
        throughput *= albedo * n_dot_l / (VX_PI_F * max(pdf, 1e-6));
    }
    path_tracer_path_state state;
    state.throughput_and_spatial_index = float4(throughput, asfloat(spatial_index));
    state.radiance = float4((float3)0.0, 0.0);
    path_state_buffer[path_index] = state;
    if (n_dot_l == 0.0)
    {
        return;
    }

    uint ray_index;
    InterlockedAdd(output_ray_count[0], 1u, ray_index);
    path_tracer_ray ray;
    ray.origin_and_path_index = float4(offset_ray(position, normal), asfloat(path_index));
    ray.direction = ray_dir;
    output_ray_buffer[ray_index] = ray;
}
