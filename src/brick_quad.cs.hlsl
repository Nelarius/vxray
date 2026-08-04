#include "brick_quad.h"

Texture3D<uint2> voxel_masks : register(t0, space0);

RWStructuredBuffer<uint4> visible_faces : register(u0, space1);
RWStructuredBuffer<uint4> draw_args : register(u1, space1);

ConstantBuffer<brick_quad_uniforms> uniforms : register(b0, space2);

bool is_occupied(int3 const brick)
{
    if (any(brick < (int3)0) || any(brick >= (int)uniforms.brick_grid_ext))
    {
        return false;
    }

    return any(voxel_masks.Load(int4(brick, 0)).rg != (uint2)0u);
}

void emit_face(uint3 const brick, uint const face)
{
    uint face_index;
    InterlockedAdd(draw_args[0].y, 1u, face_index);
    if (face_index < uniforms.face_capacity)
    {
        visible_faces[face_index] = uint4(brick, face);
    }
}

[numthreads(64, 1, 1)] void main(uint3 const dispatch_thread_id : SV_DispatchThreadID) {
    uint const brick_grid_ext = uniforms.brick_grid_ext;
    uint const total_bricks = brick_grid_ext * brick_grid_ext * brick_grid_ext;
    uint const brick_index = dispatch_thread_id.x;
    if (brick_index >= total_bricks)
    {
        return;
    }

    uint3 const brick =
        uint3(brick_index % brick_grid_ext, (brick_index / brick_grid_ext) % brick_grid_ext,
              brick_index / (brick_grid_ext * brick_grid_ext));
    if (!is_occupied(int3(brick)))
    {
        return;
    }

    int3 const neighbor_offsets[6] = {
        int3(-1, 0, 0), int3(1, 0, 0), int3(0, -1, 0), int3(0, 1, 0), int3(0, 0, -1), int3(0, 0, 1),
    };
    float3 const normals[6] = {
        float3(-1.0, 0.0, 0.0), float3(1.0, 0.0, 0.0),  float3(0.0, -1.0, 0.0),
        float3(0.0, 1.0, 0.0),  float3(0.0, 0.0, -1.0), float3(0.0, 0.0, 1.0),
    };

    float3 const brick_min = float3(brick * VX_BRICK_EXT);
    for (uint face = 0u; face < 6u; ++face)
    {
        if (is_occupied(int3(brick) + neighbor_offsets[face]))
        {
            continue;
        }

        float3 const face_center =
            brick_min + (float3)(VX_BRICK_EXT * 0.5) + normals[face] * (VX_BRICK_EXT * 0.5);
        if (dot(normals[face], uniforms.camera_position.xyz - face_center) > 0.0)
        {
            emit_face(brick, face);
        }
    }
}
