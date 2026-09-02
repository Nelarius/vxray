#include "sky.h"

struct ps_input
{
    float4 position : SV_Position;
    float2 uv : TEXCOORD0;
};

ConstantBuffer<sky_view_uniforms> uniforms : register(b0, space3);

float4 main(ps_input const input) : SV_Target0
{
    float3 const ray_start = uniforms.view_position.xyz;
    float3 const ray_dir = sky_view_uv_to_ray_dir(input.uv);
    float        ray_length = 1e38;
    float2 const planet_hit = sky_planet_intersection(ray_start, ray_dir);
    if (planet_hit.x > 0.0)
    {
        ray_length = min(ray_length, planet_hit.x);
    }

    float3 const radiance = sky_integrate_scattering(
        ray_start, ray_dir, ray_length, uniforms.sun_direction.xyz, uniforms.sun_color.rgb);
    return float4(radiance, 1.0);
}
