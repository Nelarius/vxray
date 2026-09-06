#include "constants.h"
#include "rtao.h"

#include "spatial_hash.hlsli"

struct ps_input
{
    float4 position : SV_Position;
};

Texture2D<float>         depth_tex : register(t0, space2);
Texture2D<uint>          hash_index_tex : register(t1, space2);
Texture2D<uint>          normal_tex : register(t2, space2);
Texture3D<uint>          voxel_masks : register(t3, space2);
Texture3D<uint>          voxel_aadf : register(t4, space2);
RWStructuredBuffer<uint> hash_payloads : register(u5, space2);

SamplerState depth_sampler : register(s0, space2);

ConstantBuffer<rtao_uniforms> uniforms : register(b0, space3);

float2 r2_sequence(float const n)
{
    // 2-dimensional golden ratio additive recurrence sequence
    // https://extremelearning.com.au/unreasonable-effectiveness-of-quasirandom-sequences/
    float const a1 = 0.7548777f;
    float const a2 = 0.5698403f;
    return frac(float2(n * a1, n * a2));
}

uint mask_linear_idx(int16_t3 const coord)
{
    int16_t3 const local = coord & (VX_MASK_EXT - 1);
    return local.x + local.y * VX_MASK_EXT + local.z * VX_MASK_EXT * VX_MASK_EXT;
}

bool sparse_ray_march(float3 const ray_origin, float3 const ray_dir, float const t_max)
{
    float3 const inv_dir = 1.0 / (ray_dir + (float3)(ray_dir == 0.0) * 1e-30);
    int16_t3     ipos = int16_t3(floor(ray_origin));
    float3       local_pos = ray_origin - float3(ipos);
    float        distance = 0.0;

    for (int i = 0; i < 3 * uniforms.grid_ext; ++i)
    {
        if (any((uint16_t3)ipos >= (uint16_t)uniforms.grid_ext))
        {
            return false;
        }

        {
            uint const mask = voxel_masks.Load(int4(ipos / VX_MASK_EXT, 0)).r;
            uint const idx = mask_linear_idx(ipos);
            if ((mask & (1u << idx)) != 0u)
            {
                return true;
            }
        }

        uint const     aadf = voxel_aadf.Load(int4(ipos, 0)).r;
        uint const     shift_x = ray_dir.x < 0.0 ? 0u : 5u;
        uint const     shift_y = ray_dir.y < 0.0 ? 10u : 15u;
        uint const     shift_z = ray_dir.z < 0.0 ? 20u : 25u;
        int16_t3 const bounds =
            int16_t3((aadf >> shift_x) & 31u, (aadf >> shift_y) & 31u, (aadf >> shift_z) & 31u);

        int16_t3 const cell_min = max(ipos - int16_t3(bounds), (int16_t3)0);
        int16_t3 const cell_max =
            min(ipos + 1 + int16_t3(bounds), (int16_t3)(int16_t)uniforms.grid_ext);
        float3 const exit_plane = float3(ray_dir.x < 0.0 ? cell_min.x : cell_max.x,
                                         ray_dir.y < 0.0 ? cell_min.y : cell_max.y,
                                         ray_dir.z < 0.0 ? cell_min.z : cell_max.z);

        float3 side_dist = (exit_plane - float3(ipos) - local_pos) * inv_dir;
        side_dist.x = ray_dir.x == 0.0 ? 3e+38 : side_dist.x;
        side_dist.y = ray_dir.y == 0.0 ? 3e+38 : side_dist.y;
        side_dist.z = ray_dir.z == 0.0 ? 3e+38 : side_dist.z;
        float const t = min(side_dist.x, min(side_dist.y, side_dist.z));
        distance += t;
        if (distance > t_max)
        {
            return false;
        }

        float3 const   crossed = step(side_dist, t);
        float3 const   advanced = local_pos + t * ray_dir;
        int16_t3 const cell_delta = int16_t3(floor(advanced + crossed * sign(ray_dir) * 0.5));
        ipos += cell_delta;
        local_pos = advanced - float3(cell_delta);
    }

    return false;
}

float main(ps_input const input) : SV_Target0
{
    uint width, height;
    depth_tex.GetDimensions(width, height);
    uint2 const  pixel = min(uint2(input.position.xy), uint2(width - 1u, height - 1u));
    float2 const uv = (float2(pixel) + 0.5) / float2(width, height);
    float const  depth = depth_tex.SampleLevel(depth_sampler, uv, 0.0).r;
    if (depth >= 1.0)
    {
        return -1.0;
    }

    uint const index = hash_index_tex.Load(int3(pixel, 0)).r;
    if (index == 0xFFFFFFFFu)
    {
        return -1.0;
    }

    float3 const normal = unpack_normal(normal_tex.Load(int3(pixel, 0)).r);
    float3 const face_position =
        reconstruct_position(uniforms.inverse_view_projection, uv, depth, normal);
    uint const payload = hash_payloads[index];
    uint const sample_count = payload & 0xFFFFu;
    if (sample_count >= VX_RTAO_SAMPLE_LIMIT)
    {
        return 1.0 - (float)(payload >> 16u) / (float)sample_count;
    }

    float3 const pos = offset_ray(face_position, normal);
    uint2 const  frame_seed =
        uint2(uniforms.frame_index * 0x9E3779B9u, pcg(uniforms.frame_index ^ 0xA511E9B3u));
    float2 const pixel_noise = as_normalized_float(pcg2d(pixel ^ frame_seed));
    float2 const u = frac(pixel_noise + r2_sequence((float)uniforms.sample_index));
    float3 const direction =
        orient_axis_aligned_sample_direction(sample_cosine_weighted_hemisphere(u), normal);
    uint const occlusion = sparse_ray_march(pos, direction, uniforms.rtao_radius);

    uint ex_occlusion;
    InterlockedAdd(hash_payloads[index], (occlusion << 16u) | 1u, ex_occlusion);
    return 1.0 -
           (float)((ex_occlusion >> 16u) + occlusion) / (float)((ex_occlusion & 0xFFFFu) + 1u);
}
