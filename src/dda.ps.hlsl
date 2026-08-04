#include "constants.h"
#include "dda.h"

struct ps_input
{
    float2 uv : TEXCOORD0;
};

ConstantBuffer<dda_uniforms> uniforms : register(b0, space3);

Texture3D<uint>        voxels : register(t0, space2);
Texture3D<uint2>       voxel_masks : register(t1, space2);
StructuredBuffer<uint> palette_rgba : register(t2, space2);

float3 unpack_rgba(uint const rgba)
{
    float const r = (float)(rgba & 255u) / 255.0;
    float const g = (float)((rgba >> 8u) & 255u) / 255.0;
    float const b = (float)((rgba >> 16u) & 255u) / 255.0;
    return pow(float3(r, g, b), 2.2);
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

float aabb_entry_distance(float3 const ray_origin, float3 const inv_ray_dir, float3 const p0,
                          float3 const p1)
{
    float3 const t0 = (p0 - ray_origin) * inv_ray_dir;
    float3 const t1 = (p1 - ray_origin) * inv_ray_dir;
    float3 const lo = min(t0, t1);
    return max(max(lo.x, lo.y), max(lo.z, 0.0));
}

uint trace_brick(float3 const origin, float3 const dir, float3 const inv_dir, float3 const tdelta,
                 int3 const brick_cell, uint2 const voxel_mask)
{
    int3 const  brick_min = brick_cell * VX_BRICK_EXT;
    float const t =
        aabb_entry_distance(origin, inv_dir, float3(brick_min),
                            float3(brick_min + int3(VX_BRICK_EXT, VX_BRICK_EXT, VX_BRICK_EXT)));
    float3 const local_entry = origin + t * dir - float3(brick_min);
    int3         local_cell = clamp(int3(local_entry), int3(0, 0, 0),
                                    int3(VX_BRICK_EXT - 1, VX_BRICK_EXT - 1, VX_BRICK_EXT - 1));
    float3 const s = sign(dir);
    int3 const   step_dir = int3(s);
    float3 const next = float3(local_cell) + max(float3(step_dir), float3(0.0, 0.0, 0.0));
    float3       tnext = (next - local_entry) * inv_dir;
    tnext.x = s.x == 0.0 ? 3e+38 : tnext.x; // guard against s == 0
    tnext.y = s.y == 0.0 ? 3e+38 : tnext.y;
    tnext.z = s.z == 0.0 ? 3e+38 : tnext.z;
    for (int i = 0; i < 3 * VX_BRICK_EXT; ++i)
    {
        if (any((uint3)local_cell >= (uint)VX_BRICK_EXT))
        {
            return 0u;
        }

        uint const voxel_idx = (uint)(local_cell.x + local_cell.y * VX_BRICK_EXT +
                                      local_cell.z * VX_BRICK_EXT * VX_BRICK_EXT);
        uint const word = voxel_mask[voxel_idx >> 5u];
        uint const bit = 1u << (voxel_idx & 31u);
        if ((word & bit) != 0u)
        {
            return voxels.Load(int4(brick_min + local_cell, 0)).r;
        }

        // Branchless trick: https://www.shadertoy.com/view/4dX3zl
        // step(a, x) like a < x.
        float3 const axis_mask = step(tnext, min(tnext.yzx, tnext.zxy));
        tnext += axis_mask * tdelta;
        local_cell += int3(axis_mask) * step_dir;
    }

    return 0u;
}

// Good insight into DDA: https://news.ycombinator.com/item?id=43599990

uint dda(float3 const origin, float3 const dir)
{
    float3 const inv_dir = 1.0 / dir;
    float3 const tdelta = abs(inv_dir);

    float       tmin;
    float       tmax;
    float const ext = (float)uniforms.grid_ext;
    if (!ray_box_test(origin, inv_dir, float3(0.0, 0.0, 0.0), float3(ext, ext, ext), tmin, tmax))
    {
        return 0u;
    }

    float3 const entry = origin + tmin * dir;
    int const    brick_grid_ext = uniforms.grid_ext / VX_BRICK_EXT;
    int3 brick_cell = clamp(int3(entry / (float)VX_BRICK_EXT), int3(0, 0, 0),
                            int3(brick_grid_ext - 1, brick_grid_ext - 1, brick_grid_ext - 1));

    float3 const s = sign(dir);
    int3 const   step_dir = int3(s);
    float3 const next =
        (float3(brick_cell) + max(float3(step_dir), float3(0.0, 0.0, 0.0))) * VX_BRICK_EXT;
    float3 tnext = (next - entry) * inv_dir / (float)VX_BRICK_EXT;
    tnext.x = s.x == 0.0 ? 3e+38 : tnext.x; // guard against s == 0
    tnext.y = s.y == 0.0 ? 3e+38 : tnext.y;
    tnext.z = s.z == 0.0 ? 3e+38 : tnext.z;

    for (int i = 0; i < 3 * brick_grid_ext; ++i)
    {
        if (any((uint3)brick_cell >= (uint)brick_grid_ext))
        {
            return 0u;
        }

        uint2 const voxel_mask = voxel_masks.Load(int4(brick_cell, 0)).rg;
        if (any(voxel_mask != uint2(0u, 0u)))
        {
            uint const voxel = trace_brick(origin, dir, inv_dir, tdelta, brick_cell, voxel_mask);
            if (voxel > 0u)
            {
                return voxel;
            }
        }

        // Branchless trick: https://www.shadertoy.com/view/4dX3zl
        // step(a, x) like a < x.
        float3 const axis_mask = step(tnext, min(tnext.yzx, tnext.zxy));
        tnext += axis_mask * tdelta;
        brick_cell += int3(axis_mask) * step_dir;
    }

    return 0u;
}

float4 main(ps_input const input) : SV_Target0
{
    float2 const ndc = input.uv * float2(2.0, -2.0) + float2(-1.0, 1.0);

    float3 const dir = normalize(uniforms.camera_forward.xyz + uniforms.camera_right.xyz * ndc.x +
                                 uniforms.camera_up.xyz * ndc.y);

    uint const palette_idx = dda(uniforms.camera_pos.xyz, dir);
    if (palette_idx > 0u)
    {
        return float4(unpack_rgba(palette_rgba[palette_idx]), 1.0);
    }
    else
    {
        return float4(0.02, 0.025, 0.03, 1.0);
    }
}
