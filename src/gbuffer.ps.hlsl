#include "constants.h"
#include "gbuffer.h"

#include "shared.hlsli"

struct ps_input
{
    float4 position : SV_Position;
    float2 uv : TEXCOORD0;
};

ConstantBuffer<gbuffer_uniforms> uniforms : register(b0, space3);

Texture2D<uint>        brick_coord_tex : register(t0, space2);
Texture3D<uint>        voxel_tex : register(t1, space2);
Texture3D<uint>        brick_tex : register(t2, space2);
StructuredBuffer<uint> palette_rgba : register(t3, space2);

uint voxel_at(int16_t3 const p) { return voxel_tex.Load(int4(p, 0)).r; }

uint brick_at(int16_t3 const p) { return brick_tex.Load(int4(p, 0)).r; }

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

bool camera_is_inside_occupied_brick()
{
    float const  ext = (float)uniforms.grid_ext;
    float3 const camera_position = uniforms.camera_pos.xyz;
    if (any(camera_position < (float3)0.0) || any(camera_position >= (float3)ext))
    {
        return false;
    }

    int16_t3 const camera_brick = int16_t3(camera_position / (float)VX_BRICK_EXT);
    return brick_at(camera_brick) > 0u;
}

float aabb_entry_distance(float3 const ray_origin, float3 const inv_ray_dir, float3 const p0,
                          float3 const p1)
{
    float3 const t0 = (p0 - ray_origin) * inv_ray_dir;
    float3 const t1 = (p1 - ray_origin) * inv_ray_dir;
    float3 const lo = min(t0, t1);
    return max(max(lo.x, lo.y), max(lo.z, 0.0));
}

uint trace_brick(float3 const origin, float3 const dir, float3 const inv_dir,
                 float3 const delta_dist, int16_t3 const brick_cell)
{
    int16_t3 const brick_min = brick_cell * (int16_t)VX_BRICK_EXT;
    float const    t = aabb_entry_distance(origin, inv_dir, float3(brick_min),
                                           float3(brick_min + (int16_t3)VX_BRICK_EXT));
    float3 const   local_pos = origin + t * dir - float3(brick_min);
    int16_t3     local_cell = clamp(int16_t3(local_pos), (int16_t3)0, (int16_t3)(VX_BRICK_EXT - 1));
    float3 const ray_sign = sign(dir);
    float3 const next_pos = float3(local_cell) + max(ray_sign, (float3)0.0);
    float3       side_dist = (next_pos - local_pos) * inv_dir;
    side_dist.x = ray_sign.x == 0.0 ? 3e+38 : side_dist.x;
    side_dist.y = ray_sign.y == 0.0 ? 3e+38 : side_dist.y;
    side_dist.z = ray_sign.z == 0.0 ? 3e+38 : side_dist.z;

    int16_t3 const step_sign = int16_t3(ray_sign);
    for (;;)
    {
        if (any((uint16_t3)local_cell >= (uint16_t)VX_BRICK_EXT))
        {
            return VX_NO_CELL;
        }

        int16_t3 const cell = brick_min + local_cell;
        if (voxel_at(cell) > 0u)
        {
            return pack_voxel_cell(cell);
        }

        // Branchless trick: https://www.shadertoy.com/view/4dX3zl
        float3 const axis_mask = step(side_dist, min(side_dist.yzx, side_dist.zxy));
        side_dist += axis_mask * delta_dist;
        local_cell += int16_t3(axis_mask) * step_sign;
    }
}

uint multilevel_dda(float3 const origin, float3 const dir, int16_t3 brick_cell)
{
    // Good insight into DDA: https://news.ycombinator.com/item?id=43599990

    float3 const inv_dir = 1.0 / dir;
    float3 const delta_dist = abs(inv_dir);

    float3 const ray_sign = sign(dir);
    float3 const next_pos = (float3(brick_cell) + max(ray_sign, (float3)0.0)) * VX_BRICK_EXT;
    float3       side_dist = (next_pos - origin) * inv_dir / (float)VX_BRICK_EXT;
    side_dist.x = ray_sign.x == 0.0 ? 3e+38 : side_dist.x;
    side_dist.y = ray_sign.y == 0.0 ? 3e+38 : side_dist.y;
    side_dist.z = ray_sign.z == 0.0 ? 3e+38 : side_dist.z;

    int16_t3 const step_sign = int16_t3(ray_sign);
    int16_t const  brick_grid_ext = (int16_t)uniforms.grid_ext / (int16_t)VX_BRICK_EXT;
    for (;;)
    {
        if (any((uint16_t3)brick_cell >= (uint16_t)brick_grid_ext))
        {
            return VX_NO_CELL;
        }

        if (brick_at(brick_cell) > 0u)
        {
            uint const packed_cell = trace_brick(origin, dir, inv_dir, delta_dist, brick_cell);
            if (packed_cell != VX_NO_CELL)
            {
                return packed_cell;
            }
        }

        // Branchless trick: https://www.shadertoy.com/view/4dX3zl
        float3 const axis_mask = step(side_dist, min(side_dist.yzx, side_dist.zxy));
        side_dist += axis_mask * delta_dist;
        brick_cell += int16_t3(axis_mask) * step_sign;
    }
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
    bool const camera_inside = camera_is_inside_occupied_brick();
    uint       entry_brick_record = 0u;
    if (!camera_inside)
    {
        entry_brick_record = brick_coord_tex.Load(int3(input.position.xy, 0)).r;
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

    uint const packed_cell = multilevel_dda(trace_origin, dir, first_brick);
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
