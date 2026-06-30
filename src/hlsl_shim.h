#pragma once

#if defined(__STDC__)

#include <stdalign.h>
#include <stdint.h>

typedef uint32_t uint;

typedef struct float2
{
    union
    {
        alignas(8) float data[2];
        struct
        {
            float x, y;
        };
    };
} float2;

typedef struct float3
{
    union
    {
        alignas(16) float data[3];
        struct
        {
            float x, y, z;
        };
    };
} float3;

typedef struct float4
{
    union
    {
        alignas(16) float data[4];
        struct
        {
            float x, y, z, w;
        };
    };
} float4;

#define float2(x, y) ((float2){x, y})
#define float3(x, y, z) ((float3){x, y, z})
#define float4(x, y, z, w) ((float4){x, y, z, w})

#endif
