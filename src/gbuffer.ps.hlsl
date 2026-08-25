#include "constants.h"
#include "gbuffer.h"

#include "shared.hlsli"

struct ps_input
{
    float4 position : SV_Position;
    float2 uv : TEXCOORD0;
};

ConstantBuffer<gbuffer_uniforms> uniforms : register(b0, space3);

Texture3D<uint>        voxels : register(t0, space2);
Texture3D<uint2>       voxel_masks : register(t1, space2);
Texture3D<uint2>       brick_masks : register(t2, space2);
StructuredBuffer<uint> palette_rgba : register(t3, space2);

uint voxel_at(int16_t3 const p) { return voxels.Load(int4(p, 0)).r; }

uint pack_voxel_cell(int16_t3 const cell)
{
    uint3 const c = uint3(cell);
    return c.x | (c.y << 10u) | (c.z << 20u);
}

int16_t3 unpack_voxel_cell(uint const packed)
{
    return int16_t3(packed & 1023u, (packed >> 10u) & 1023u, (packed >> 20u) & 1023u);
}

bool ray_box_test(float3 const ray_origin, float3 const inv_ray_dir, float3 const p0,
                  float3 const p1, out float tmin, out float tmax)
{
    // "RAY AXIS-ALIGNED BOUNDING BOX INTERSECTION", Ray Tracing Gems II

    float3 const t0 = (p0 - ray_origin) * inv_ray_dir; // inf is okay here
    float3 const t1 = (p1 - ray_origin) * inv_ray_dir;
    float3 const lo = min(t0, t1);
    float3 const hi = max(t0, t1);
    tmin = max(max(lo.x, lo.y), max(lo.z, 0.0));
    tmax = min(min(hi.x, hi.y), hi.z);
    return tmin <= tmax;
}

// NOTE: ext is expected to be a power of two
int mask_linear_idx(int16_t3 const coord, int const ext)
{
    int const x = coord.x & (ext - 1);
    int const y = coord.y & (ext - 1);
    int const z = coord.z & (ext - 1);
    return x + y * ext + z * ext * ext;
}

bool mask_bit_test(uint2 const mask, int const index)
{
    uint const word = mask[index >> 5u];
    uint const bit = 1u << (index & 31u);
    return (word & bit) != 0;
}

uint sparse_ray_march(float3 const ray_origin, float3 const ray_dir)
{
    float3 const inv_dir = 1.0 / ray_dir;
    float const  grid_ext = (float)uniforms.grid_ext;
    float        tmin, tmax;
    if (!ray_box_test(ray_origin, inv_dir, (float3)0.0, (float3)grid_ext, tmin, tmax))
    {
        return VX_NO_CELL;
    }

    bool const   camera_inside = all(ray_origin >= (float3)0.0) && all(ray_origin < grid_ext);
    float3 const start =
        camera_inside ? ray_origin : offset_ray(ray_origin + tmin * ray_dir, ray_dir);
    int16_t3 ipos = int16_t3(floor(start));
    float3   local_pos = start - float3(ipos);

    for (int i = 0; i < 3 * uniforms.grid_ext; ++i)
    {
        if (any((uint16_t3)ipos >= (uint16_t)uniforms.grid_ext))
        {
            return VX_NO_CELL;
        }

        // Determine occupancy level to use

        int     mask_idx = mask_linear_idx(ipos >> 3, 4);
        uint2   mask = brick_masks.Load(int4(ipos >> 5, 0)).rg;
        int16_t scale = (int16_t)VX_BRICK_EXT;

        if (mask_bit_test(mask, mask_idx))
        {
            mask_idx = mask_linear_idx(ipos, 4);
            mask = voxel_masks.Load(int4(ipos >> 2, 0)).rg;
            scale = 1;

            if (mask_bit_test(mask, mask_idx))
            {
                return pack_voxel_cell(ipos);
            }
        }

        // Determine largest empty region containing `ipos`

        int16_t lod;

        if ((mask.x | mask.y) == 0u)
        {
            // Entire 4x4x4 region is empty
            lod = 4;
        }
        else
        {
            uint const mask_part = mask_idx < 32 ? mask.x : mask.y;
            // Is the containing 2x2x2 region empty?
            // 0x0A preserves the high bits of the 2x2x2 block's local x and y coordinates.
            // 0x00330033 selects the relevant 2x2x2 bits.
            if (((mask_part >> (mask_idx & 0x0A)) & 0x00330033u) == 0)
            {
                lod = 2;
            }
            else
            {
                lod = 1;
            }
        }

        lod *= scale;

        // Intersect the ray with the forward-facing planes of the empty region
        int16_t const  cell_mask = lod - 1;
        int16_t3 const cell_min = ipos & ~cell_mask;
        int16_t3 const cell_max = cell_min + lod;
        float3 const   exit_plane = float3(ray_dir.x < 0.0 ? cell_min.x : cell_max.x,
                                           ray_dir.y < 0.0 ? cell_min.y : cell_max.y,
                                           ray_dir.z < 0.0 ? cell_min.z : cell_max.z);

        float3 side_dist = (exit_plane - float3(ipos) - local_pos) * inv_dir;
        side_dist.x = ray_dir.x == 0.0 ? 3e+38 : side_dist.x;
        side_dist.y = ray_dir.y == 0.0 ? 3e+38 : side_dist.y;
        side_dist.z = ray_dir.z == 0.0 ? 3e+38 : side_dist.z;
        float const  t = min(side_dist.x, min(side_dist.y, side_dist.z));
        float3 const crossed = step(side_dist, t);

        float3 const   advanced = local_pos + t * ray_dir;
        int16_t3 const cell_delta = int16_t3(floor(advanced + crossed * sign(ray_dir) * 0.5));
        ipos += cell_delta;
        local_pos = advanced - float3(cell_delta);
    }

    return VX_NO_CELL;
}

struct voxel_intersection
{
    float  distance;
    float3 normal;
};

voxel_intersection intersect_voxel(float3 const origin, float3 const dir, int16_t3 const cell)
{
    float3 const inv_dir = 1.0 / dir;
    float3 const cell_min = float3(cell);
    float3 const cell_max = cell_min + 1.0;
    float3 const t0 = (cell_min - origin) * inv_dir;
    float3 const t1 = (cell_max - origin) * inv_dir;
    float3 const t_near = min(t0, t1);
    float3 const axis_mask = step(-t_near, min(-t_near.yzx, -t_near.zxy));

    // TODO: this can produce diagonal normals for an exact edge or corner hit

    voxel_intersection result;
    result.distance = max(t_near.x, max(t_near.y, t_near.z));
    result.normal = -sign(dir) * axis_mask;

    return result;
}

struct ps_output
{
    uint  albedo : SV_Target0;
    uint  normal : SV_Target1;
    float depth : SV_Depth;
};

ps_output miss()
{
    ps_output output;
    output.albedo = 0u;
    output.normal = 0u;
    output.depth = 1.0;
    return output;
}

ps_output main(ps_input const input)
{
    float2 const ndc = input.uv * float2(2.0, -2.0) + float2(-1.0, 1.0);
    float3 const dir =
        normalize(unproject(uniforms.inverse_view_projection, ndc, 1.0) - uniforms.camera_pos.xyz);

    uint const packed_cell = sparse_ray_march(uniforms.camera_pos.xyz, dir);
    if (packed_cell == VX_NO_CELL)
    {
        return miss();
    }

    int16_t3 const           cell = unpack_voxel_cell(packed_cell);
    voxel_intersection const intersection = intersect_voxel(uniforms.camera_pos.xyz, dir, cell);
    if (intersection.distance <= 0.0)
    {
        return miss();
    }

    float3 const hit_position = uniforms.camera_pos.xyz + intersection.distance * dir;
    float4 const clip_position = mul(uniforms.view_projection, float4(hit_position, 1.0));

    ps_output output;
    output.albedo = palette_rgba[voxel_at(cell)];
    output.normal = pack_normal(intersection.normal);
    output.depth = clip_position.z / clip_position.w;
    return output;
}
