#include "path_tracer.h"
#include "sky.h"

#include "sharc.hlsli"

Texture2D<float> depth_tex : register(t0, space0);
Texture2D<uint>  albedo_tex : register(t1, space0);
Texture2D<uint>  normal_tex : register(t2, space0);
SamplerState     depth_sampler : register(s0, space0);

RWStructuredBuffer<path_tracer_ray>        output_ray_buffer : register(u0, space1);
RWStructuredBuffer<path_tracer_path_state> path_state_buffer : register(u1, space1);
RWStructuredBuffer<uint>                   output_ray_count : register(u2, space1);
RWStructuredBuffer<uint>                   hash_checksums : register(u3, space1);
RWStructuredBuffer<uint4>                  sharc_accumulation : register(u4, space1);
RWStructuredBuffer<uint>                   hash_frames : register(u5, space1);
RWStructuredBuffer<float4>                 path_trace_output : register(u6, space1);

ConstantBuffer<path_tracer_uniforms> uniforms : register(b0, space2);

[numthreads(VX_WAVEFRONT_SCREEN_THREAD_COUNT, VX_WAVEFRONT_SCREEN_THREAD_COUNT, 1)] void
main(uint2 const tid : SV_DispatchThreadID) {
    uint width, height;
    depth_tex.GetDimensions(width, height);

    uint2 pixel = tid;
    if (uniforms.mode == VX_PATH_TRACE_MODE_UPDATE)
    {
        uint const  k = VX_PATH_TRACE_UPDATE_TILE_SIZE;
        uint2 const tile = tid;
        uint const  phase = pcg(tile.x ^ pcg(tile.y)) % (k * k);
        uint const  slot = (uniforms.frame % (k * k) + phase) % (k * k);
        pixel = tile * k + uint2(slot % k, slot / k);
    }
    if (pixel.x >= width || pixel.y >= height)
    {
        return;
    }

    uint const   path_index = pixel.y * width + pixel.x;
    float2 const uv = (float2(pixel) + 0.5) / float2(width, height);
    float const  depth = depth_tex.SampleLevel(depth_sampler, uv, 0.0).r;
    if (depth >= 1.0)
    {
        return;
    }

    float3 const normal = unpack_normal(normal_tex.Load(int3(pixel, 0)).r);
    float3 const position =
        reconstruct_position(uniforms.inverse_view_projection, uv, depth, normal);
    float3 const albedo = unpack_albedo(albedo_tex.Load(int3(pixel, 0)).r).rgb;

    path_tracer_path_state state = (path_tracer_path_state)0;
    if (uniforms.mode == VX_PATH_TRACE_MODE_UPDATE)
    {
        spatial_hash_key const key = sharc_key(position, normal, uniforms);
        uint const             cache_index =
            sharc_find_or_insert(key, uniforms.frame, hash_checksums, hash_frames);
        if (cache_index == VX_SPATIAL_HASH_INVALID_INDEX)
        {
            return;
        }
        sharc_append_vertex(state, cache_index, albedo);
    }

    uint const   stream_id = path_index ^ (uniforms.mode * 0x9E3779B9u);
    float3 const samples = halton_sample_3d(uniforms.frame, 0u, stream_id);
    float2 const u = samples.xy;
    bool const   sample_sun = samples.z < 0.5;
    float const  cos_theta_max = cos(VX_SKY_SOLAR_RADIUS_RAD);
    float3 const local_dir =
        sample_sun ? sample_cone(u, cos_theta_max) : sample_cosine_weighted_hemisphere(u);
    float3 const ray_dir = sample_sun
                               ? orient_sample_direction(local_dir, uniforms.sun_direction.xyz)
                               : orient_axis_aligned_sample_direction(local_dir, normal);
    float const  n_dot_l = max(dot(normal, ray_dir), 0.0);
    if (n_dot_l == 0.0)
    {
        if (uniforms.mode == VX_PATH_TRACE_MODE_UPDATE)
        {
            sharc_accumulate(state, (float3)0.0, sharc_accumulation);
        }
        else
        {
            path_trace_output[path_index] = (float4)0.0;
        }
        return;
    }

    float const  pdf = 0.5 * (pdf_cone(dot(uniforms.sun_direction.xyz, ray_dir), cos_theta_max) +
                              pdf_cosine_weighted_hemisphere(n_dot_l));
    float3 const throughput = albedo * n_dot_l / (VX_PI_F * max(pdf, 1e-6));
    if (uniforms.mode == VX_PATH_TRACE_MODE_UPDATE)
    {
        sharc_multiply_weights(state, throughput);
    }
    else
    {
        state.throughput_and_path_length.xyz = throughput;
    }
    path_state_buffer[path_index] = state;

    uint ray_index;
    InterlockedAdd(output_ray_count[0], 1u, ray_index);
    path_tracer_ray ray;
    ray.origin_and_path_index = float4(offset_ray(position, normal), asfloat(path_index));
    ray.direction = ray_dir;
    output_ray_buffer[ray_index] = ray;
}
