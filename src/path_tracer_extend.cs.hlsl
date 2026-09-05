#include "path_tracer.h"
#include "sky.h"

#include "shared.hlsli"

Texture2D<float4>                 sky_view_tex : register(t0, space0);
Texture3D<uint>                   voxels : register(t1, space0);
Texture3D<uint>                   voxel_masks : register(t2, space0);
Texture3D<uint>                   brick_masks : register(t3, space0);
Texture3D<uint>                   chunk_masks : register(t4, space0);
Texture3D<uint>                   voxel_aadf : register(t5, space0);
Texture3D<uint>                   brick_aadf : register(t6, space0);
Texture3D<uint>                   chunk_aadf : register(t7, space0);
StructuredBuffer<uint>            palette_rgba : register(t8, space0);
StructuredBuffer<path_tracer_ray> input_ray_buffer : register(t9, space0);
StructuredBuffer<uint>            input_ray_count : register(t10, space0);
SamplerState                      sky_view_sampler : register(s0, space0);

RWStructuredBuffer<path_tracer_ray>        output_ray_buffer : register(u0, space1);
RWStructuredBuffer<uint>                   output_ray_count : register(u1, space1);
RWStructuredBuffer<path_tracer_path_state> path_state_buffer : register(u2, space1);

ConstantBuffer<path_tracer_uniforms> uniforms : register(b0, space2);

uint mask_linear_idx(int16_t3 const coord)
{
    int16_t3 const local = coord & (VX_MASK_EXT - 1);
    return local.x + local.y * VX_MASK_EXT + local.z * VX_MASK_EXT * VX_MASK_EXT;
}

bool mask_bit_test(uint const mask, int16_t3 const coord)
{
    return (mask & (1u << mask_linear_idx(coord))) != 0u;
}

bool chunk_occupied(int16_t3 const coord)
{
    return mask_bit_test(chunk_masks.Load(int4(coord / VX_MASK_EXT, 0)).r, coord);
}

bool brick_occupied(int16_t3 const coord)
{
    return mask_bit_test(brick_masks.Load(int4(coord / VX_MASK_EXT, 0)).r, coord);
}

bool voxel_occupied(int16_t3 const coord)
{
    return mask_bit_test(voxel_masks.Load(int4(coord / VX_MASK_EXT, 0)).r, coord);
}

bool ray_box_test(float3 const ray_origin, float3 const inv_ray_dir, float3 const p0,
                  float3 const p1, out float tmin, out float tmax, out float3 entry_mask)
{
    float3 const t0 = (p0 - ray_origin) * inv_ray_dir;
    float3 const t1 = (p1 - ray_origin) * inv_ray_dir;
    float3 const lo = min(t0, t1);
    float3 const hi = max(t0, t1);
    tmin = max(max(lo.x, lo.y), max(lo.z, 0.0));
    tmax = min(min(hi.x, hi.y), hi.z);
    entry_mask = tmin > 0.0 ? step(tmin, lo) : (float3)0.0;
    return tmin <= tmax;
}

float3 aadf_exit_plane(int16_t3 const coord, int16_t const cell_ext, uint const packed,
                       float3 const ray_dir)
{
    uint const     shift_x = ray_dir.x < 0.0 ? 0u : 5u;
    uint const     shift_y = ray_dir.y < 0.0 ? 10u : 15u;
    uint const     shift_z = ray_dir.z < 0.0 ? 20u : 25u;
    int16_t3 const bounds =
        int16_t3((packed >> shift_x) & 31u, (packed >> shift_y) & 31u, (packed >> shift_z) & 31u);
    int16_t3 const lower = max((coord - int16_t3(bounds)) * cell_ext, (int16_t3)0);
    int16_t3 const upper =
        min((coord + 1 + int16_t3(bounds)) * cell_ext, (int16_t3)(int16_t)uniforms.grid_ext);
    return float3(ray_dir.x < 0.0 ? lower.x : upper.x, ray_dir.y < 0.0 ? lower.y : upper.y,
                  ray_dir.z < 0.0 ? lower.z : upper.z);
}

uint pack_voxel_cell(int16_t3 const cell, uint const axis)
{
    uint3 const c = uint3(cell);
    return c.x | (c.y << 10u) | (c.z << 20u) | (axis << 30u);
}

int16_t3 unpack_voxel_cell(uint const packed)
{
    return int16_t3(packed & 1023u, (packed >> 10u) & 1023u, (packed >> 20u) & 1023u);
}

uint sparse_ray_march(float3 const ray_origin, float3 const ray_dir)
{
    float3 const inv_dir = 1.0 / (ray_dir + (float3)(ray_dir == 0.0) * 1e-30);
    float        tmin;
    float        tmax;
    float3       entry_mask;
    if (!ray_box_test(ray_origin, inv_dir, (float3)0.0, (float3)(float)uniforms.grid_ext, tmin,
                      tmax, entry_mask))
    {
        return VX_NO_CELL;
    }

    float3 const start = ray_origin + tmin * ray_dir;
    int16_t3     ipos = int16_t3(floor(start + entry_mask * sign(ray_dir) * 0.5));
    float3       local_pos = start - float3(ipos);
    float3 const entry_weights = entry_mask * abs(ray_dir);
    uint         crossed_axis = entry_weights.y > entry_weights.x ? 1u : 0u;
    crossed_axis = entry_weights.z > max(entry_weights.x, entry_weights.y) ? 2u : crossed_axis;

    for (int i = 0; i < 3 * uniforms.grid_ext; ++i)
    {
        if (any((uint16_t3)ipos >= (uint16_t)uniforms.grid_ext))
        {
            return VX_NO_CELL;
        }

        int16_t  cell_ext = VX_BRICK_EXT * VX_CHUNK_EXT;
        int16_t3 coord = ipos / (VX_BRICK_EXT * VX_CHUNK_EXT);
        bool     occupied = chunk_occupied(coord);
        if (occupied)
        {
            cell_ext = VX_BRICK_EXT;
            coord = ipos / VX_BRICK_EXT;
            occupied = brick_occupied(coord);
        }
        if (occupied)
        {
            cell_ext = 1;
            coord = ipos;
            occupied = voxel_occupied(coord);
        }
        if (occupied)
        {
            return pack_voxel_cell(ipos, crossed_axis);
        }

        uint aadf;
        if (cell_ext == VX_BRICK_EXT * VX_CHUNK_EXT)
        {
            aadf = chunk_aadf.Load(int4(coord, 0)).r;
        }
        else if (cell_ext == VX_BRICK_EXT)
        {
            aadf = brick_aadf.Load(int4(coord, 0)).r;
        }
        else
        {
            aadf = voxel_aadf.Load(int4(coord, 0)).r;
        }

        float3 const exit_plane = aadf_exit_plane(coord, cell_ext, aadf, ray_dir);
        float3       side_dist = (exit_plane - float3(ipos) - local_pos) * inv_dir;
        side_dist.x = ray_dir.x == 0.0 ? 3e+38 : side_dist.x;
        side_dist.y = ray_dir.y == 0.0 ? 3e+38 : side_dist.y;
        side_dist.z = ray_dir.z == 0.0 ? 3e+38 : side_dist.z;
        float const  t = min(side_dist.x, min(side_dist.y, side_dist.z));
        float3 const crossed = step(side_dist, t);
        float3 const crossed_weights = crossed * abs(ray_dir);
        crossed_axis = crossed_weights.y > crossed_weights.x ? 1u : 0u;
        crossed_axis =
            crossed_weights.z > max(crossed_weights.x, crossed_weights.y) ? 2u : crossed_axis;
        float3 const   advanced = local_pos + max(t, 0.0001) * ray_dir;
        int16_t3 const cell_delta = int16_t3(floor(advanced + crossed * sign(ray_dir) * 0.5));
        ipos += cell_delta;
        local_pos = advanced - float3(cell_delta);
    }
    return VX_NO_CELL;
}

float3 sky_radiance(float3 const ray_dir, float3 const sun_color)
{
    float2 const sky_uv = sky_view_ray_dir_to_uv(ray_dir);
    float3       radiance = sky_view_tex.SampleLevel(sky_view_sampler, sky_uv, 0.0).rgb;
    float const  cos_theta_max = cos(VX_SKY_SOLAR_RADIUS_RAD);
    if (dot(ray_dir, uniforms.sun_direction.xyz) >= cos_theta_max)
    {
        radiance += sun_color / sqrt(1.0 - cos_theta_max);
    }
    return radiance;
}

[numthreads(VX_WAVEFRONT_EXTEND_THREAD_COUNT, 1, 1)] void
main(uint const thread_id : SV_DispatchThreadID) {
    if (thread_id >= input_ray_count[0])
    {
        return;
    }

    path_tracer_ray const  ray = input_ray_buffer[thread_id];
    float3 const           ray_origin = ray.origin_and_path_index.xyz;
    float3 const           ray_dir = ray.direction;
    uint const             path_index = asuint(ray.origin_and_path_index.w);
    path_tracer_path_state state = path_state_buffer[path_index];
    float3                 throughput = state.throughput_and_spatial_index.xyz;

    uint const packed_cell = sparse_ray_march(ray_origin, ray_dir);

    // Shade miss

    if (packed_cell == VX_NO_CELL)
    {
        state.radiance.xyz +=
            throughput * sky_radiance(ray_dir, uniforms.transmitted_sun_color.rgb);
        path_state_buffer[path_index] = state;
        return;
    }

    // Shade intersection

    int16_t3 const cell = unpack_voxel_cell(packed_cell);
    uint const     axis_index = packed_cell >> 30u;
    float3 const   axis_mask = float3(axis_index == uint3(0u, 1u, 2u));
    float3 const   normal = -sign(ray_dir) * axis_mask;
    float3 const   entry_plane = float3(cell) + (float3)(ray_dir < 0.0);
    float const    distance = dot(entry_plane - ray_origin, axis_mask) / dot(ray_dir, axis_mask);
    float3 const   position = ray_origin + distance * ray_dir;
    float3 const   albedo = unpack_albedo(palette_rgba[voxels.Load(int4(cell, 0)).r]).rgb;

    // Generate next direction (MIS, albertian and sun disk)

    uint const   seed = pcg(uniforms.frame + pcg(uniforms.bounce));
    uint2 const  hash = pcg2d(uint2(path_index, seed));
    float2 const u = as_normalized_float(hash);
    bool const   sample_sun = pcg(path_index + seed) < 0x80000000u;
    float const  cos_theta_max = cos(VX_SKY_SOLAR_RADIUS_RAD);
    float3 const local_dir =
        sample_sun ? sample_cone(u, cos_theta_max) : sample_cosine_weighted_hemisphere(u);
    float3 const next_ray_dir = sample_sun
                                    ? orient_sample_direction(local_dir, uniforms.sun_direction.xyz)
                                    : orient_axis_aligned_sample_direction(local_dir, normal);
    float const  n_dot_l = max(dot(normal, next_ray_dir), 0.0);
    if (n_dot_l == 0.0)
    {
        return;
    }
    float const pdf =
        0.5 * (pdf_cone(dot(uniforms.sun_direction.xyz, next_ray_dir), cos_theta_max) +
               pdf_cosine_weighted_hemisphere(n_dot_l));
    throughput *= albedo * n_dot_l / (VX_PI_F * max(pdf, 1e-6));
    state.throughput_and_spatial_index.xyz = throughput;
    path_state_buffer[path_index] = state;

    uint output_ray_index;
    InterlockedAdd(output_ray_count[0], 1u, output_ray_index);
    path_tracer_ray output_ray;
    output_ray.origin_and_path_index = float4(offset_ray(position, normal), asfloat(path_index));
    output_ray.direction = next_ray_dir;
    output_ray_buffer[output_ray_index] = output_ray;
}
