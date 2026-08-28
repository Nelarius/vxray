#include "constants.h"

#define VX_PARENT_EXT VX_BRICK_EXT
#define VX_PARENT_CELL_COUNT (VX_PARENT_EXT * VX_PARENT_EXT * VX_PARENT_EXT)
#define VX_OCCUPIED_BIT 0x80000000u

Texture3D<uint>   occupancy_masks : register(t0, space0);
RWTexture3D<uint> aadf : register(u0, space1);

cbuffer Uniforms : register(b0, space2) { uint grid_ext; };

groupshared uint cached[VX_PARENT_CELL_COUNT];

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

void grow_bound(inout uint current, uint const neighbour, uint const direction)
{
    if ((neighbour & VX_OCCUPIED_BIT) != 0u)
    {
        return;
    }

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

[numthreads(VX_PARENT_EXT, VX_PARENT_EXT, VX_PARENT_EXT)] void
main(uint3 const local_id : SV_GroupThreadID, uint3 const group_id : SV_GroupID,
     uint const local_index : SV_GroupIndex) {
    uint3 const pos = group_id * VX_PARENT_EXT + local_id;
    bool const  valid = all(pos < grid_ext);
    uint        current = VX_OCCUPIED_BIT;
    if (valid)
    {
        current = cell_occupied(pos) ? VX_OCCUPIED_BIT : 0u;
    }
    cached[local_index] = current;
    GroupMemoryBarrierWithGroupSync();

    [unroll] for (uint round = 0u; round < VX_PARENT_EXT - 1u; ++round)
    {
        if ((current & VX_OCCUPIED_BIT) == 0u)
        {
            if (local_id.x > 0u)
            {
                grow_bound(current, cached[local_index - 1u], 0u);
            }
            if (local_id.x + 1u < VX_PARENT_EXT)
            {
                grow_bound(current, cached[local_index + 1u], 1u);
            }
        }
        GroupMemoryBarrierWithGroupSync();
        cached[local_index] = current;
        GroupMemoryBarrierWithGroupSync();

        if ((current & VX_OCCUPIED_BIT) == 0u)
        {
            if (local_id.y > 0u)
            {
                grow_bound(current, cached[local_index - VX_PARENT_EXT], 2u);
            }
            if (local_id.y + 1u < VX_PARENT_EXT)
            {
                grow_bound(current, cached[local_index + VX_PARENT_EXT], 3u);
            }
        }
        GroupMemoryBarrierWithGroupSync();
        cached[local_index] = current;
        GroupMemoryBarrierWithGroupSync();

        if ((current & VX_OCCUPIED_BIT) == 0u)
        {
            if (local_id.z > 0u)
            {
                grow_bound(current, cached[local_index - VX_PARENT_EXT * VX_PARENT_EXT], 4u);
            }
            if (local_id.z + 1u < VX_PARENT_EXT)
            {
                grow_bound(current, cached[local_index + VX_PARENT_EXT * VX_PARENT_EXT], 5u);
            }
        }
        GroupMemoryBarrierWithGroupSync();
        cached[local_index] = current;
        GroupMemoryBarrierWithGroupSync();
    }

    if (valid)
    {
        aadf[pos] = current & ~VX_OCCUPIED_BIT;
    }
}
