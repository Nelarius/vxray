#include "rtao.h"

#include "spatial_hash.hlsli"

struct ps_input
{
    float4 position : SV_Position;
};

Texture2D<float> visibility_tex : register(t0, space2);
Texture2D<float> depth_tex : register(t1, space2);
Texture2D<uint>  normal_tex : register(t2, space2);

ConstantBuffer<rtao_filter_uniforms> uniforms : register(b0, space3);

static float const CACHE_FAILURE_VISIBILITY = -1.0;
static float const EPSILON = 1e-3;

float main(ps_input const input) : SV_Target0
{
    uint width, height;
    visibility_tex.GetDimensions(width, height);
    uint2 const pixel = min(uint2(input.position.xy), uint2(width - 1u, height - 1u));
    float const visibility_c = visibility_tex.Load(int3(pixel, 0)).r;
    float const device_depth_c = depth_tex.Load(int3(pixel, 0)).r;
    if (visibility_c <= CACHE_FAILURE_VISIBILITY || device_depth_c >= 1.0)
    {
        return visibility_c;
    }

    float3 const normal_c = unpack_normal(normal_tex.Load(int3(pixel, 0)).r);

    // Calculating the depth gradient based on the view-space normals:
    // ∇z = (nx / √(1 - nx²), ny / √(1 - ny²))
    //
    // Source: https://chrismile.net/blog/2024/svgf-nabla-depth/
    //
    // Unlike a difference estimate, this estimate of the gradient is not neighborhood-based and
    // yields much better results at the edges of a surface.
    // float2 const grad_z = svgf_depth_gradient(normal_c, params.view_mat);
    float3 const n = mul((float3x3)uniforms.view_matrix, normal_c);
    float const  dzdx = n.x / (sqrt(max(1.0 - n.x * n.x, 0.0)) + EPSILON);
    float const  dzdy = n.y / (sqrt(max(1.0 - n.y * n.y, 0.0)) + EPSILON);
    float2 const grad_z = float2(dzdx, dzdy);
    float const  depth_c =
        linear_view_depth(device_depth_c, uniforms.near_plane, uniforms.far_plane);

    static float const kernel[3] = {3.0 / 8.0, 1.0 / 4.0, 1.0 / 16.0};
    float              visibility_sum = 0.0;
    float              weight_sum = 0.0;

    for (int dy = -2; dy <= 2; ++dy)
    {
        for (int dx = -2; dx <= 2; ++dx)
        {
            int2 const offset = int2(dx, dy) * (int)uniforms.step_width;
            int2 const neighbor = int2(pixel) + offset;
            if (any(neighbor < 0) || any(neighbor >= int2(width, height)))
            {
                continue;
            }

            float const visibility_n = visibility_tex.Load(int3(neighbor, 0)).r;
            float const device_depth_n = depth_tex.Load(int3(neighbor, 0)).r;
            if (visibility_n <= CACHE_FAILURE_VISIBILITY || device_depth_n >= 1.0)
            {
                continue;
            }

            // Normal weight: n(p) · n(q)
            float3 const normal_n = unpack_normal(normal_tex.Load(int3(neighbor, 0)).r);
            float const  normal_weight = max(0.0, dot(normal_c, normal_n));

            // Depth weight: exp(-|z(p) - z(q)| / (sigma_z * |∇z(p) · (p - q)| + e))
            float const depth_n =
                linear_view_depth(device_depth_n, uniforms.near_plane, uniforms.far_plane);
            float const depth_delta = abs(depth_c - depth_n);
            float const projected_depth_delta = abs(dot(grad_z, float2(offset)));
            float const depth_weight =
                exp(-depth_delta /
                    ((float)uniforms.step_width * uniforms.sigma_depth * projected_depth_delta +
                     EPSILON));

            float const filter_weight = kernel[abs(dx)] * kernel[abs(dy)];
            float const weight = normal_weight * depth_weight * filter_weight;
            visibility_sum += visibility_n * weight;
            weight_sum += weight;
        }
    }

    return weight_sum > EPSILON ? visibility_sum / weight_sum : visibility_c;
}
