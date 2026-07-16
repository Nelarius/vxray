#include "dda.h"

struct ps_input
{
    float2 uv : TEXCOORD0;
};

ConstantBuffer<dda_uniforms> uniforms : register(b0, space3);

Texture3D<uint>        voxels : register(t0, space2);
StructuredBuffer<uint> palette_rgba : register(t1, space2);

float3 unpack_rgba(uint const rgba)
{
    float const r = (float)(rgba & 255u) / 255.0;
    float const g = (float)((rgba >> 8u) & 255u) / 255.0;
    float const b = (float)((rgba >> 16u) & 255u) / 255.0;
    return pow(float3(r, g, b), 2.2);
}

uint voxel_at(int3 const p) { return voxels.Load(int4(p, 0)).r; }

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

// Good insight into DDA: https://news.ycombinator.com/item?id=43599990

uint dda(float3 const origin, float3 const dir)
{
    float3 const inv_dir = 1.0 / dir;
    float3 const inv_abs_dir = abs(inv_dir);

    // Ray-bounds test

    float tmin;
    float tmax;
    {
        float const  ext = (float)uniforms.grid_ext;
        float3 const bmin = float3(0.0, 0.0, 0.0);
        float3 const bmax = float3(ext, ext, ext);
        if (!ray_box_test(origin, 1.0 / dir, bmin, bmax, tmin, tmax))
        {
            return 0u;
        }
    }

    // Calculate the number of steps along the ray

    float3 const entry = origin + tmin * dir;
    int const    max_idx = uniforms.grid_ext - 1;
    int3 const   min_cell = int3(0, 0, 0);
    int3 const   max_cell = int3(max_idx, max_idx, max_idx);
    int3 const   start_cell = clamp(int3(entry), min_cell, max_cell);

    // Initialize traversal vectors

    float3 const s = sign(dir);
    int3 const   step_dir = int3(s);
    float3 const next = float3(start_cell) + max(float3(step_dir), float3(0.0, 0.0, 0.0));
    int3         cell = start_cell;
    float3       tnext = (next - entry) * inv_dir;
    tnext.x = s.x == 0.0 ? 3e+38 : tnext.x; // guard against s == 0
    tnext.y = s.y == 0.0 ? 3e+38 : tnext.y;
    tnext.z = s.z == 0.0 ? 3e+38 : tnext.z;
    float3 const tdelta = inv_abs_dir;

    // Trace voxels

    for (int i = 0; i < 3 * uniforms.grid_ext; ++i)
    {
        if (any(cell < min_cell) || any(cell > max_cell))
        {
            return 0u;
        }

        uint const v = voxel_at(cell);
        if (v > 0u)
        {
            return v;
        }

        // Branchless trick: https://www.shadertoy.com/view/4dX3zl
        // step(a, x) like a < x.
        float3 const axis_mask = step(tnext, min(tnext.yzx, tnext.zxy));
        tnext += axis_mask * tdelta;
        cell += int3(axis_mask) * step_dir;
    }

    return 0u;
}

float4 main(ps_input const input) : SV_Target0
{
    float2 const pixel = input.uv * uniforms.viewport.xy;
    float2 const ndc = float2((pixel.x / uniforms.viewport.x) * 2.0 - 1.0,
                              1.0 - (pixel.y / uniforms.viewport.y) * 2.0);

    float const aspect = uniforms.viewport.x / uniforms.viewport.y;
    float const fov = radians(uniforms.viewport.z);
    float const scale = tan(fov * 0.5);

    float3 const dir =
        normalize(uniforms.camera_forward.xyz + uniforms.camera_right.xyz * ndc.x * aspect * scale +
                  uniforms.camera_up.xyz * ndc.y * scale);

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
