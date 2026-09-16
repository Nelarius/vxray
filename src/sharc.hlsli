#pragma once

#include "path_tracer.h"

#include "spatial_hash.hlsli"

float sharc_cell_size(float3 const position, float3 const camera_position, float const sp,
                      float const smin, float const vertical_fov, uint const render_height)
{
    float const camera_distance = max(length(position - camera_position), smin);
    return compute_cell_size(camera_distance, vertical_fov, render_height, sp, smin);
}

spatial_hash_key sharc_key(float3 const position, float3 const normal,
                           path_tracer_uniforms const uniforms)
{
    float const cell_size =
        sharc_cell_size(position, uniforms.camera_pos.xyz, uniforms.sp, uniforms.smin,
                        uniforms.vertical_fov, uniforms.render_height);
    return make_spatial_hash_key(position, normal, cell_size);
}

uint sharc_find_or_insert(spatial_hash_key const key, uint const frame,
                          RWStructuredBuffer<uint> hash_checksums,
                          RWStructuredBuffer<uint> hash_frames)
{
    for (uint attempt = 0u; attempt < 2u; ++attempt)
    {
        uint index = key.hash & VX_PATH_TRACE_SPATIAL_HASH_MASK;
        uint empty_index = VX_SPATIAL_HASH_INVALID_INDEX;
        for (uint probe = 0u; probe < VX_PATH_TRACE_SPATIAL_HASH_PROBE_COUNT; ++probe)
        {
            uint const checksum = hash_checksums[index];
            if (checksum == key.checksum)
            {
                uint ex_frame;
                InterlockedExchange(hash_frames[index], frame, ex_frame);
                return index;
            }
            if (checksum == 0u && empty_index == VX_SPATIAL_HASH_INVALID_INDEX)
            {
                empty_index = index;
            }
            index = (index + 1u) & VX_PATH_TRACE_SPATIAL_HASH_MASK;
        }

        if (empty_index == VX_SPATIAL_HASH_INVALID_INDEX)
        {
            return VX_SPATIAL_HASH_INVALID_INDEX;
        }

        uint ex_checksum;
        InterlockedCompareExchange(hash_checksums[empty_index], 0u, key.checksum, ex_checksum);
        if (ex_checksum == 0u || ex_checksum == key.checksum)
        {
            uint ex_frame;
            InterlockedExchange(hash_frames[empty_index], frame, ex_frame);
            return empty_index;
        }
    }
    return VX_SPATIAL_HASH_INVALID_INDEX;
}

uint sharc_find(spatial_hash_key const key, RWStructuredBuffer<uint> hash_checksums)
{
    uint index = key.hash & VX_PATH_TRACE_SPATIAL_HASH_MASK;
    for (uint probe = 0u; probe < VX_PATH_TRACE_SPATIAL_HASH_PROBE_COUNT; ++probe)
    {
        uint const checksum = hash_checksums[index];
        if (checksum == key.checksum)
        {
            return index;
        }
        index = (index + 1u) & VX_PATH_TRACE_SPATIAL_HASH_MASK;
    }
    return VX_SPATIAL_HASH_INVALID_INDEX;
}

float3 sharc_inverse_albedo(float3 const albedo)
{
    return float3(albedo.x > 1e-6 ? rcp(albedo.x) : 0.0, albedo.y > 1e-6 ? rcp(albedo.y) : 0.0,
                  albedo.z > 1e-6 ? rcp(albedo.z) : 0.0);
}

void sharc_append_vertex(inout path_tracer_path_state state, uint const cache_index,
                         float3 const albedo)
{
    uint const path_length = asuint(state.throughput_and_path_length.w);
    uint const last = min(path_length, VX_SHARC_PROPAGATION_DEPTH - 1u);
    for (uint i = last; i > 0u; --i)
    {
        state.sharc_vertices[i] = state.sharc_vertices[i - 1u];
    }
    state.sharc_vertices[0] = float4(sharc_inverse_albedo(albedo), asfloat(cache_index));
    state.throughput_and_path_length.w = asfloat(min(path_length + 1u, VX_SHARC_PROPAGATION_DEPTH));
}

void sharc_multiply_weights(inout path_tracer_path_state state, float3 const throughput)
{
    uint const path_length = asuint(state.throughput_and_path_length.w);
    for (uint i = 0u; i < path_length; ++i)
    {
        state.sharc_vertices[i].xyz *= throughput;
    }
}

void sharc_accumulate(path_tracer_path_state const state, float3 const radiance,
                      RWStructuredBuffer<uint4> accumulation)
{
    uint const path_length = asuint(state.throughput_and_path_length.w);
    for (uint i = 0u; i < path_length; ++i)
    {
        float3 const sample = radiance * state.sharc_vertices[i].xyz;
        uint3 const  encoded = encode_fixed_point(sample, VX_PATH_TRACE_MAX_SAMPLE_SHADING,
                                                  VX_PATH_TRACE_ACCUMULATION_SCALE);
        uint const   cache_index = asuint(state.sharc_vertices[i].w);
        uint         ex;
        InterlockedAdd(accumulation[cache_index].x, encoded.x, ex);
        InterlockedAdd(accumulation[cache_index].y, encoded.y, ex);
        InterlockedAdd(accumulation[cache_index].z, encoded.z, ex);
        InterlockedAdd(accumulation[cache_index].w, 1u, ex);
    }
}

float3 sharc_resolved_shading(uint4 const payload)
{
    return decode_fixed_point(payload.xyz, VX_PATH_TRACE_MAX_SAMPLE_SHADING,
                              VX_PATH_TRACE_ACCUMULATION_SCALE);
}
