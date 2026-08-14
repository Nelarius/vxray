#include "constants.h"
#include "gbuffer.h"

#include "shared.hlsli"

struct ps_input
{
    float2 uv : TEXCOORD0;
};

ConstantBuffer<gbuffer_uniforms> uniforms : register(b0, space3);

Texture2D<float> entry_depth : register(t0, space2);
SamplerState     entry_sampler : register(s0, space2);

Texture3D<uint>        voxels : register(t1, space2);
Texture3D<uint>        bricks : register(t2, space2);
StructuredBuffer<uint> palette_rgba : register(t3, space2);

uint voxel_at(int16_t3 const p) { return voxels.Load(int4(p, 0)).r; }

uint brick_at(int16_t3 const p) { return bricks.Load(int4(p, 0)).r; }

uint pack_voxel_cell(int16_t3 const cell)
{
    uint3 const c = uint3(cell);
    return c.x | (c.y << 10u) | (c.z << 20u);
}

int16_t3 unpack_voxel_cell(uint const packed)
{
    return int16_t3(packed & 1023u, (packed >> 10u) & 1023u, (packed >> 20u) & 1023u);
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

float3 reconstruct_brick_pos(float3 const reconstructed_pos, float3 const dir)
{
    float const  brick_ext = (float)VX_BRICK_EXT;
    float3 const brick_pos = reconstructed_pos / brick_ext;
    float3 const rounded_brick_pos = round(brick_pos);
    float3 const plane_distance = abs(brick_pos - rounded_brick_pos);

    float3 face_axis;
    if (plane_distance.x <= plane_distance.y && plane_distance.x <= plane_distance.z)
    {
        face_axis = float3(1.0, 0.0, 0.0);
    }
    else if (plane_distance.y <= plane_distance.z)
    {
        face_axis = float3(0.0, 1.0, 0.0);
    }
    else
    {
        face_axis = float3(0.0, 0.0, 1.0);
    }

    float3 const face_pos = lerp(reconstructed_pos, rounded_brick_pos * brick_ext, face_axis);
    float3 const inward_normal = face_axis * sign(dir);
    return offset_ray(face_pos, inward_normal);
}

uint trace_brick(float3 const origin, float3 const dir, float3 const inv_dir, float3 const tdelta,
                 int16_t3 const brick_cell)
{
    int16_t3 const brick_min = brick_cell * (int16_t)VX_BRICK_EXT;
    float const    t = aabb_entry_distance(origin, inv_dir, float3(brick_min),
                                           float3(brick_min + (int16_t3)VX_BRICK_EXT));
    float3 const   local_entry = origin + t * dir - float3(brick_min);
    int16_t3 local_cell = clamp(int16_t3(local_entry), (int16_t3)0, (int16_t3)(VX_BRICK_EXT - 1));
    float3 const   s = sign(dir);
    int16_t3 const step_dir = int16_t3(s);
    float3 const   next = float3(local_cell) + max(float3(step_dir), (float3)0.0);
    float3         tnext = (next - local_entry) * inv_dir;
    tnext.x = s.x == 0.0 ? 3e+38 : tnext.x;
    tnext.y = s.y == 0.0 ? 3e+38 : tnext.y;
    tnext.z = s.z == 0.0 ? 3e+38 : tnext.z;
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
        float3 const axis_mask = step(tnext, min(tnext.yzx, tnext.zxy));
        tnext += axis_mask * tdelta;
        local_cell += int16_t3(axis_mask) * step_dir;
    }
}

// The `entry` (i.e. entrypoint to DDA) is expected to be a point contained in the grid.
uint multilevel_dda(float3 const entry, float3 const dir)
{
    // Good insight into DDA: https://news.ycombinator.com/item?id=43599990

    float3 const inv_dir = 1.0 / dir;
    float3 const tdelta = abs(inv_dir);

    float3 const   s = sign(dir);
    int16_t3 const step_dir = int16_t3(s);
    int16_t3       brick_cell = int16_t3(entry / (float)VX_BRICK_EXT);
    float3 const   next = (float3(brick_cell) + max(float3(step_dir), (float3)0.0)) * VX_BRICK_EXT;
    float3         tnext = (next - entry) * inv_dir / (float)VX_BRICK_EXT;
    tnext.x = s.x == 0.0 ? 3e+38 : tnext.x;
    tnext.y = s.y == 0.0 ? 3e+38 : tnext.y;
    tnext.z = s.z == 0.0 ? 3e+38 : tnext.z;

    int16_t const brick_grid_ext = (int16_t)uniforms.grid_ext / (int16_t)VX_BRICK_EXT;
    for (;;)
    {
        if (any((uint16_t3)brick_cell >= (uint16_t)brick_grid_ext))
        {
            return VX_NO_CELL;
        }

        if (brick_at(brick_cell) > 0u)
        {
            uint const packed_cell = trace_brick(entry, dir, inv_dir, tdelta, brick_cell);
            if (packed_cell != VX_NO_CELL)
            {
                return packed_cell;
            }
        }

        // Branchless trick: https://www.shadertoy.com/view/4dX3zl
        float3 const axis_mask = step(tnext, min(tnext.yzx, tnext.zxy));
        tnext += axis_mask * tdelta;
        brick_cell += int16_t3(axis_mask) * step_dir;
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
    float const entry_device_depth = entry_depth.SampleLevel(entry_sampler, input.uv, 0.0).r;
    bool const  camera_inside = camera_is_inside_occupied_brick();
    if (entry_device_depth >= 1.0 && !camera_inside)
    {
        return miss();
    }

    float2 const ndc = input.uv * float2(2.0, -2.0) + float2(-1.0, 1.0);
    float3 const dir =
        normalize(unproject(uniforms.inverse_view_projection, ndc, 1.0) - uniforms.camera_pos.xyz);
    float3 const trace_origin =
        camera_inside
            ? offset_ray(uniforms.camera_pos.xyz, dir)
            : reconstruct_brick_pos(
                  unproject(uniforms.inverse_view_projection, ndc, entry_device_depth), dir);

    uint const packed_cell = multilevel_dda(trace_origin, dir);
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
