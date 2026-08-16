#include "brick_quad.h"

StructuredBuffer<uint> visible_faces : register(t0, space0);

ConstantBuffer<brick_quad_uniforms> uniforms : register(b0, space1);

struct vs_output
{
    float4               position : SV_Position;
    nointerpolation uint packed_brick : TEXCOORD0;
};

float3 face_corner(uint const face, float2 const corner)
{
    float const edge = (float)VX_BRICK_EXT;
    if (face == 0u)
    {
        return float3(0.0, corner.x, corner.y) * edge;
    }
    if (face == 1u)
    {
        return float3(1.0, corner.x, corner.y) * edge;
    }
    if (face == 2u)
    {
        return float3(corner.x, 0.0, corner.y) * edge;
    }
    if (face == 3u)
    {
        return float3(corner.x, 1.0, corner.y) * edge;
    }
    if (face == 4u)
    {
        return float3(corner.x, corner.y, 0.0) * edge;
    }
    return float3(corner.x, corner.y, 1.0) * edge;
}

vs_output main(uint const vertex_id : SV_VertexID, uint const instance_id : SV_InstanceID)
{
    uint const   corner_indices[6] = {0u, 1u, 2u, 0u, 2u, 3u};
    float2 const corners[4] = {
        float2(0.0, 0.0),
        float2(1.0, 0.0),
        float2(1.0, 1.0),
        float2(0.0, 1.0),
    };

    uint const  face_data = visible_faces[instance_id];
    uint3 const brick =
        uint3(face_data & 0xffu, (face_data >> 8u) & 0xffu, (face_data >> 16u) & 0xffu);
    uint const   face = (face_data >> 24u) & 0x7u;
    float3 const world_position =
        float3(brick * VX_BRICK_EXT) + face_corner(face, corners[corner_indices[vertex_id]]);

    vs_output output;
    output.position = mul(uniforms.view_projection, float4(world_position, 1.0));
    output.packed_brick = face_data & 0x00ffffffu;
    return output;
}
