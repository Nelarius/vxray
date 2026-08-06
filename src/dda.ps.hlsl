#include "constants.h"
#include "dda.h"

struct ps_input
{
    float2 uv : TEXCOORD0;
};

ConstantBuffer<dda_uniforms> uniforms : register(b0, space3);

Texture2D<float> entry_depth : register(t0, space2);
SamplerState     entry_sampler : register(s0, space2);

Texture3D<uint>        voxels : register(t1, space2);
Texture3D<uint>        bricks : register(t2, space2);
StructuredBuffer<uint> palette_rgba : register(t3, space2);

float3 unpack_rgba(uint const rgba)
{
    float const r = (float)(rgba & 255u) / 255.0;
    float const g = (float)((rgba >> 8u) & 255u) / 255.0;
    float const b = (float)((rgba >> 16u) & 255u) / 255.0;
    return pow(float3(r, g, b), 2.2);
}

uint voxel_at(int3 const p) { return voxels.Load(int4(p, 0)).r; }

uint brick_at(int3 const p) { return bricks.Load(int4(p, 0)).r; }

float3 offset_ray(float3 const p, float3 const n)
{
    // "A Fast and Robust Method for Avoiding Self-Intersection", Ray Tracing Gems
    float const int_scale = 256.0;
    float const float_scale = 1e-5;
    float const origin = 1e-3;

    int3 const   offset = int3(int_scale * n);
    float3 const po = float3(asfloat(asint(p.x) + ((p.x < 0.0) ? -offset.x : offset.x)),
                             asfloat(asint(p.y) + ((p.y < 0.0) ? -offset.y : offset.y)),
                             asfloat(asint(p.z) + ((p.z < 0.0) ? -offset.z : offset.z)));

    return float3((abs(p.x) < origin) ? p.x + float_scale * n.x : po.x,
                  (abs(p.y) < origin) ? p.y + float_scale * n.y : po.y,
                  (abs(p.z) < origin) ? p.z + float_scale * n.z : po.z);
}

bool camera_is_inside_occupied_brick()
{
    float const  ext = (float)uniforms.grid_ext;
    float3 const camera_position = uniforms.camera_pos.xyz;
    if (any(camera_position < (float3)0.0) || any(camera_position >= (float3)ext))
    {
        return false;
    }

    int3 const camera_brick = int3(camera_position / (float)VX_BRICK_EXT);
    return brick_at(camera_brick) > 0u;
}

float3 unproject(float2 const ndc, float const depth)
{
    float4 const world = mul(uniforms.inverse_view_projection, float4(ndc, depth, 1.0));
    return world.xyz / world.w;
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
                 int3 const brick_cell)
{
    int3 const   brick_min = brick_cell * VX_BRICK_EXT;
    float const  t = aabb_entry_distance(origin, inv_dir, float3(brick_min),
                                         float3(brick_min + (int3)VX_BRICK_EXT));
    float3 const local_entry = origin + t * dir - float3(brick_min);
    int3         local_cell = clamp(int3(local_entry), (int3)0, (int3)(VX_BRICK_EXT - 1));
    float3 const s = sign(dir);
    int3 const   step_dir = int3(s);
    float3 const next = float3(local_cell) + max(float3(step_dir), (float3)0.0);
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

        uint const v = voxel_at(brick_min + local_cell);
        if (v > 0u)
        {
            return v;
        }

        if (tnext.x < tnext.y)
        {
            if (tnext.x < tnext.z)
            {
                local_cell.x += step_dir.x;
                tnext.x += tdelta.x;
            }
            else
            {
                local_cell.z += step_dir.z;
                tnext.z += tdelta.z;
            }
        }
        else if (tnext.y < tnext.z)
        {
            local_cell.y += step_dir.y;
            tnext.y += tdelta.y;
        }
        else
        {
            local_cell.z += step_dir.z;
            tnext.z += tdelta.z;
        }
    }

    return 0u;
}

// Good insight into DDA: https://news.ycombinator.com/item?id=43599990

uint multilevel_dda(float3 const origin, float3 const dir)
{
    float3 const inv_dir = 1.0 / dir;
    float3 const tdelta = abs(inv_dir);

    float3 const entry = origin;
    int const    brick_grid_ext = uniforms.grid_ext / VX_BRICK_EXT;
    int3         brick_cell = int3(entry / (float)VX_BRICK_EXT);

    float3 const s = sign(dir);
    int3 const   step_dir = int3(s);
    float3 const next = (float3(brick_cell) + max(float3(step_dir), (float3)0.0)) * VX_BRICK_EXT;
    float3       tnext = (next - entry) * inv_dir / (float)VX_BRICK_EXT;
    tnext.x = s.x == 0.0 ? 3e+38 : tnext.x; // guard against s == 0
    tnext.y = s.y == 0.0 ? 3e+38 : tnext.y;
    tnext.z = s.z == 0.0 ? 3e+38 : tnext.z;

    for (int i = 0; i < 3 * brick_grid_ext; ++i)
    {
        if (any((uint3)brick_cell >= (uint)brick_grid_ext))
        {
            return 0u;
        }

        if (brick_at(brick_cell) > 0u)
        {
            uint const voxel = trace_brick(origin, dir, inv_dir, tdelta, brick_cell);
            if (voxel > 0u)
            {
                return voxel;
            }
        }

        float3 const axis_mask = step(tnext, min(tnext.yzx, tnext.zxy));
        tnext += axis_mask * tdelta;
        brick_cell += int3(axis_mask) * step_dir;
    }

    return 0u;
}

float4 main(ps_input const input) : SV_Target0
{
    float const depth = entry_depth.SampleLevel(entry_sampler, input.uv, 0.0).r;
    bool const  camera_inside = camera_is_inside_occupied_brick();
    if (depth >= 1.0 && !camera_inside)
    {
        return float4(0.02, 0.025, 0.03, 1.0);
    }

    float2 const ndc = input.uv * float2(2.0, -2.0) + float2(-1.0, 1.0);
    float3 const dir = normalize(unproject(ndc, 1.0) - uniforms.camera_pos.xyz);
    float3 const trace_origin = camera_inside ? offset_ray(uniforms.camera_pos.xyz, dir)
                                              : offset_ray(unproject(ndc, depth), dir);

    uint const palette_idx = multilevel_dda(trace_origin, dir);
    if (palette_idx > 0u)
    {
        return float4(unpack_rgba(palette_rgba[palette_idx]), 1.0);
    }
    else
    {
        return float4(0.02, 0.025, 0.03, 1.0);
    }
}
