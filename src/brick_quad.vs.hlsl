#include "brick_quad.h"

StructuredBuffer<uint4> visible_faces : register(t0, space0);

ConstantBuffer<brick_quad_uniforms> uniforms : register(b0, space1);

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

float4 main(uint const vertex_id : SV_VertexID, uint const instance_id : SV_InstanceID)
    : SV_Position
{
    uint const   corner_indices[6] = {0u, 1u, 2u, 0u, 2u, 3u};
    float2 const corners[4] = {
        float2(0.0, 0.0),
        float2(1.0, 0.0),
        float2(1.0, 1.0),
        float2(0.0, 1.0),
    };

    uint4 const  face_record = visible_faces[instance_id];
    float3 const world_position = float3(face_record.xyz * VX_BRICK_EXT) +
                                  face_corner(face_record.w, corners[corner_indices[vertex_id]]);
    float3 const relative = world_position - uniforms.camera_position.xyz;
    float const  view_x = dot(relative, uniforms.camera_right.xyz);
    float const  view_y = dot(relative, uniforms.camera_up.xyz);
    float const  view_z = dot(relative, uniforms.camera_forward.xyz);
    float const  depth_scale = uniforms.far_plane / (uniforms.far_plane - uniforms.near_plane);
    float const  clip_z = depth_scale * view_z - uniforms.near_plane * uniforms.far_plane /
                                                     (uniforms.far_plane - uniforms.near_plane);

    return float4(view_x / uniforms.projection_scale.x, view_y / uniforms.projection_scale.y,
                  clip_z, view_z);
}
