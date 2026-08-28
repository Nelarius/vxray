#include "constants.h"

Texture3D<uint>   occupancy_masks : register(t0, space0);
Texture3D<uint>   source_aadf : register(t1, space0);
RWTexture3D<uint> destination_aadf : register(u0, space1);

cbuffer Uniforms : register(b0, space2)
{
    uint grid_ext;
    uint axis;
    uint initialize;
};

bool cell_occupied(uint3 const pos)
{
    uint3 const mask_pos = pos / 2u;
    uint3 const local = pos & 1u;
    uint const  bit = local.x | (local.y << 1u) | (local.z << 2u);
    return (occupancy_masks.Load(int4(mask_pos, 0)).r & (1u << bit)) != 0u;
}

uint aadf_bound(uint const packed, uint const direction)
{
    return (packed >> (5u * direction)) & 31u;
}

void grow_bound(inout uint current, uint3 const neighbour_pos, uint const direction)
{
    if (cell_occupied(neighbour_pos))
    {
        return;
    }

    uint const neighbour = source_aadf.Load(int4(neighbour_pos, 0)).r;
    uint const opposite = direction ^ 1u;
    [unroll] for (uint bound_direction = 0u; bound_direction < 6u; ++bound_direction)
    {
        if (bound_direction != opposite &&
            aadf_bound(neighbour, bound_direction) < aadf_bound(current, bound_direction))
        {
            return;
        }
    }

    current += 1u << (5u * direction);
}

[numthreads(VX_BRICK_EXT, VX_BRICK_EXT, VX_BRICK_EXT)] void
main(uint3 const pos : SV_DispatchThreadID) {
    if (any(pos >= grid_ext))
    {
        return;
    }
    if (initialize != 0u || cell_occupied(pos))
    {
        destination_aadf[pos] = 0u;
        return;
    }

    uint        current = source_aadf.Load(int4(pos, 0)).r;
    uint const  negative_direction = axis * 2u;
    uint const  positive_direction = negative_direction + 1u;
    uint3 const direction = uint3(axis == 0u, axis == 1u, axis == 2u);

    if (pos[axis] > 0u && aadf_bound(current, negative_direction) < 31u)
    {
        grow_bound(current, pos - direction, negative_direction);
    }
    if (pos[axis] + 1u < grid_ext && aadf_bound(current, positive_direction) < 31u)
    {
        grow_bound(current, pos + direction, positive_direction);
    }

    destination_aadf[pos] = current;
}
