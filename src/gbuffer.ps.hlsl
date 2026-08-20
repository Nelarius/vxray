#include "constants.h"
#include "gbuffer.h"

#include "shared.hlsli"

struct ps_input
{
    float4 position : SV_Position;
    float2 uv : TEXCOORD0;
};

ConstantBuffer<gbuffer_uniforms> uniforms : register(b0, space3);

Texture2D<uint>        entry_bricks : register(t0, space2);
Texture3D<uint>        voxels : register(t1, space2);
Texture3D<uint2>       voxel_masks : register(t2, space2);
Texture3D<uint2>       brick_masks : register(t3, space2);
StructuredBuffer<uint> palette_rgba : register(t4, space2);

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

int16_t3 unpack_brick_cell(uint const packed)
{
    return int16_t3(packed & 255u, (packed >> 8u) & 255u, (packed >> 16u) & 255u);
}

float aabb_entry_distance(float3 const ray_origin, float3 const inv_ray_dir, float3 const p0,
                          float3 const p1)
{
    float3 const t0 = (p0 - ray_origin) * inv_ray_dir;
    float3 const t1 = (p1 - ray_origin) * inv_ray_dir;
    float3 const lo = min(t0, t1);
    return max(max(lo.x, lo.y), max(lo.z, 0.0));
}

// NOTE: ext is expected to be a power of two
int mask_linear_idx(int16_t3 const coord, int const ext)
{
    int x = coord.x & (ext - 1);
    int y = coord.y & (ext - 1);
    int z = coord.z & (ext - 1);
    return x + y * ext + z * ext * ext;
}

bool mask_bit_test(uint2 const mask, int index)
{
    uint const word = mask[index >> 5u];
    uint const bit = 1u << (index & 31u);
    return (word & bit) != 0;
}

uint sparse_ray_march(float3 const ray_origin, float3 const ray_dir,
                      int16_t3 const entry_brick_cell)
{
    float3 const inv_dir = 1.0 / ray_dir;
    float3 const t_start = (step(0.0, ray_dir) - ray_origin) * inv_dir;

    int16_t3 ipos;
    {
        int3 const brick_min = int3(entry_brick_cell) * VX_BRICK_EXT;
        int3 const brick_max = brick_min + (int3)VX_BRICK_EXT;
        float t = aabb_entry_distance(ray_origin, inv_dir, float3(brick_min), float3(brick_max));
        t = asfloat(asuint(t) + 5u); // bias forward slightly. TODO: offset using face normal
        float3 const pos = ray_origin + t * ray_dir;
        ipos = int16_t3(floor(pos));
    }

    for (int i = 0; i < 256; ++i)
    {
        if (any((uint16_t3)ipos >= (uint16_t)uniforms.grid_ext))
        {
            return VX_NO_CELL;
        }

        // Determine occupancy level to use

        int   mask_idx = mask_linear_idx(ipos >> 3, 4);
        uint2 mask = brick_masks.Load(int4(ipos >> 5, 0)).rg;
        int   scale = VX_BRICK_EXT;

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

        int lod;

        if ((mask.x | mask.y) == 0)
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

        // Move ipos to the far edge of the empty region in the direction of travel

        int const cell_mask = lod - 1;
        // & ~cell_mask: truncate to beginning of empty region
        // | cell_mask: fill in to end of empty region
        ipos.x = (int16_t)(ray_dir.x < 0.0 ? (ipos.x & ~cell_mask) : (ipos.x | cell_mask));
        ipos.y = (int16_t)(ray_dir.y < 0.0 ? (ipos.y & ~cell_mask) : (ipos.y | cell_mask));
        ipos.z = (int16_t)(ray_dir.z < 0.0 ? (ipos.z & ~cell_mask) : (ipos.z | cell_mask));

        // Intersect the ray with the forward-facing planes of the new cell

        float3 side_dist = t_start + float3(ipos) * inv_dir;
        side_dist.x = ray_dir.x == 0.0 ? 3e+38 : side_dist.x;
        side_dist.y = ray_dir.y == 0.0 ? 3e+38 : side_dist.y;
        side_dist.z = ray_dir.z == 0.0 ? 3e+38 : side_dist.z;
        float t = min(side_dist.x, min(side_dist.y, side_dist.z));
        t = asfloat(asuint(t) + 5u); // bias forward slightly. TODO: offset using face normal
        float3 const next_pos = ray_origin + t * ray_dir;
        ipos = int16_t3(floor(next_pos));
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

bool brick_is_occupied(int16_t3 const p)
{
    uint2 const mask = brick_masks.Load(int4(p >> 2, 0)).rg;
    return mask_bit_test(mask, mask_linear_idx(p, 4));
}

bool camera_is_inside_occupied_brick()
{
    float const  ext = (float)uniforms.grid_ext;
    float3 const camera_position = uniforms.camera_pos.xyz;
    if (any(camera_position < (float3)0.0) || any(camera_position >= (float3)ext))
    {
        return false;
    }

    int16_t3 const camera_brick = int16_t3(camera_position / (float)VX_BRICK_EXT);
    return brick_is_occupied(camera_brick);
}

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
    bool const camera_inside = camera_is_inside_occupied_brick();
    uint       entry_brick_record = 0u;
    if (!camera_inside)
    {
        entry_brick_record = entry_bricks.Load(int3(input.position.xy, 0)).r;
        if (entry_brick_record == 0u)
        {
            return miss();
        }
    }

    float2 const ndc = input.uv * float2(2.0, -2.0) + float2(-1.0, 1.0);
    float3 const dir =
        normalize(unproject(uniforms.inverse_view_projection, ndc, 1.0) - uniforms.camera_pos.xyz);
    float3 const trace_origin =
        camera_inside ? offset_ray(uniforms.camera_pos.xyz, dir) : uniforms.camera_pos.xyz;
    int16_t3 const first_brick = camera_inside ? int16_t3(trace_origin / (float)VX_BRICK_EXT)
                                               : unpack_brick_cell(entry_brick_record - 1u);

    uint const packed_cell = sparse_ray_march(trace_origin, dir, first_brick);
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
