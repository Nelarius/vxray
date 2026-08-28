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
Texture3D<uint>        voxel_masks : register(t1, space2);
Texture3D<uint>        brick_masks : register(t2, space2);
Texture3D<uint>        chunk_masks : register(t3, space2);
Texture3D<uint>        voxel_aadf : register(t4, space2);
Texture3D<uint>        brick_aadf : register(t5, space2);
Texture3D<uint>        chunk_aadf : register(t6, space2);
StructuredBuffer<uint> palette_rgba : register(t7, space2);

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

    float3 const t0 = (p0 - ray_origin) * inv_ray_dir;
    float3 const t1 = (p1 - ray_origin) * inv_ray_dir;
    float3 const lo = min(t0, t1);
    float3 const hi = max(t0, t1);
    tmin = max(max(lo.x, lo.y), max(lo.z, 0.0));
    tmax = min(min(hi.x, hi.y), hi.z);
    return tmin <= tmax;
}

uint mask_linear_idx(int16_t3 const coord)
{
    int16_t3 const local = coord & (VX_MASK_EXT - 1);
    return local.x + local.y * VX_MASK_EXT + local.z * VX_MASK_EXT * VX_MASK_EXT;
}

bool mask_bit_test(uint const mask, int16_t3 const coord)
{
    uint const idx = mask_linear_idx(coord);
    return (mask & (1u << idx)) != 0u;
}

bool chunk_occupied(int16_t3 const coord)
{
    uint const mask = chunk_masks.Load(int4(coord / VX_MASK_EXT, 0)).r;
    return mask_bit_test(mask, coord);
}

bool brick_occupied(int16_t3 const coord)
{
    uint const mask = brick_masks.Load(int4(coord / VX_MASK_EXT, 0)).r;
    return mask_bit_test(mask, coord);
}

bool voxel_occupied(int16_t3 const coord)
{
    uint const mask = voxel_masks.Load(int4(coord / VX_MASK_EXT, 0)).r;
    return mask_bit_test(mask, coord);
}

// Convert an packed AADF element into the forward-facing planes of the empty region.
float3 aadf_exit_plane(int16_t3 const coord, int16_t const cell_ext, uint const packed,
                       float3 const ray_dir)
{
    // Axis-aligned distance fields are packed:
    //
    //  0 ..  4  -X
    //  5 ..  9  +X
    // 10 .. 14  -Y
    // 15 .. 19  +Y
    // 20 .. 24  -Z
    // 25 .. 29  +Z
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

uint sparse_ray_march(float3 const ray_origin, float3 const ray_dir)
{
    float3 const inv_dir = 1.0 / (ray_dir + (float3)(ray_dir == 0.0) * 1e-30);
    float const  grid_ext = (float)uniforms.grid_ext;
    float        tmin;
    float        tmax;
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
            return pack_voxel_cell(ipos);
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
        float const    t = min(side_dist.x, min(side_dist.y, side_dist.z));
        float3 const   crossed = step(side_dist, t);
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
    float3 const inv_dir = 1.0 / (dir + (float3)(dir == 0.0) * 1e-30);
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
