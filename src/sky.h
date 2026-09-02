#pragma once

#include "hlsl_shim.h"

#define VX_SKY_LUT_WIDTH 400
#define VX_SKY_LUT_HEIGHT 400
#define VX_SKY_SUN_COLOR float4(100.f, 100.f, 100.f, 0.f)

typedef struct sky_view_uniforms
{
    float4 view_position;
    float4 sun_direction;
    float4 sun_color;
} sky_view_uniforms;

#if defined(__HLSL_VERSION)

#define VX_SKY_PLANET_RADIUS 6371e3
#define VX_SKY_PLANET_CENTER float3(0.0, -VX_SKY_PLANET_RADIUS, 0.0)
#define VX_SKY_HEIGHT 100e3
#define VX_SKY_RAYLEIGH_HEIGHT (VX_SKY_HEIGHT * 0.08)
#define VX_SKY_MIE_HEIGHT (VX_SKY_HEIGHT * 0.012)
#define VX_SKY_RAYLEIGH_S float3(5.802e-6, 13.558e-6, 33.100e-6)
#define VX_SKY_MIE_S float3(3.996e-6, 3.996e-6, 3.996e-6)
#define VX_SKY_MIE_A float3(4.40e-6, 4.40e-6, 4.40e-6)
#define VX_SKY_OZONE_A float3(0.650e-6, 1.881e-6, 0.085e-6)
#define VX_SKY_PI 3.14159265358979323846

float2 sky_sphere_intersection(float3 const ray_start, float3 const ray_dir,
                               float3 const sphere_center, float const sphere_radius)
{
    float3 const oc = ray_start - sphere_center;
    float const  a = dot(ray_dir, oc);
    float const  b = dot(oc, oc) - sphere_radius * sphere_radius;
    float const  discriminant = a * a - b;
    if (discriminant > 0.0)
    {
        float const x = sqrt(discriminant);
        return float2(-a - x, -a + x);
    }
    return float2(-1.0, -1.0);
}

float2 sky_atmosphere_intersection(float3 const ray_start, float3 const ray_dir)
{
    return sky_sphere_intersection(ray_start, ray_dir, VX_SKY_PLANET_CENTER,
                                   VX_SKY_PLANET_RADIUS + VX_SKY_HEIGHT);
}

float2 sky_planet_intersection(float3 const ray_start, float3 const ray_dir)
{
    return sky_sphere_intersection(ray_start, ray_dir, VX_SKY_PLANET_CENTER, VX_SKY_PLANET_RADIUS);
}

float sky_rayleigh_phase(float const cos_theta)
{
    return 3.0 * (1.0 + cos_theta * cos_theta) / (16.0 * VX_SKY_PI);
}

float sky_mie_phase(float const cos_theta)
{
    float const g = 0.8;
    float const k = 1.55 * g - 0.55 * g * g * g;
    float const k_cos_theta = k * cos_theta;
    return (1.0 - k * k) / ((4.0 * VX_SKY_PI) * (1.0 - k_cos_theta) * (1.0 - k_cos_theta));
}

float sky_atmosphere_height(float3 const position)
{
    return length(position - VX_SKY_PLANET_CENTER) - VX_SKY_PLANET_RADIUS;
}

float3 sky_atmosphere_density(float const height)
{
    float const rayleigh = exp(-max(0.0, height / VX_SKY_RAYLEIGH_HEIGHT));
    float const mie = exp(-max(0.0, height / VX_SKY_MIE_HEIGHT));
    float const ozone = max(0.0, 1.0 - abs(height - 25000.0) / 15000.0);
    return float3(rayleigh, mie, ozone);
}

float3 sky_integrate_optical_depth(float3 const ray_start, float3 const ray_dir)
{
    float2 const intersection = sky_atmosphere_intersection(ray_start, ray_dir);
    int const    sample_count = 8;
    float const  step_size = intersection.y / (float)sample_count;

    float3 optical_depth = (float3)0.0;
    for (int i = 0; i < sample_count; ++i)
    {
        float3 const p = ray_start + ray_dir * ((float)i + 0.5) * step_size;
        optical_depth += sky_atmosphere_density(sky_atmosphere_height(p)) * step_size;
    }
    return optical_depth;
}

float3 sky_transmittance(float3 const optical_depth)
{
    float3 const rayleigh = VX_SKY_RAYLEIGH_S * optical_depth.x;
    float3 const mie = (VX_SKY_MIE_S + VX_SKY_MIE_A) * optical_depth.y;
    float3 const ozone = VX_SKY_OZONE_A * optical_depth.z;
    return exp(-(rayleigh + mie + ozone));
}

float3 sky_integrate_scattering(float3 const ray_start, float3 const ray_dir,
                                float const ray_length, float3 const light_dir,
                                float3 const light_color)
{
    float const ray_height = sky_atmosphere_height(ray_start);
    float const sample_distribution_exponent =
        1.0 + saturate(1.0 - ray_height / VX_SKY_HEIGHT) * 8.0;

    float2 const atmosphere_hit = sky_atmosphere_intersection(ray_start, ray_dir);
    float        final_ray_length = min(ray_length, atmosphere_hit.y);
    float3       current_ray_start = ray_start;
    if (atmosphere_hit.x > 0.0)
    {
        current_ray_start += ray_dir * atmosphere_hit.x;
        final_ray_length -= atmosphere_hit.x;
    }

    float const cos_theta = dot(ray_dir, light_dir);
    float const phase_r = sky_rayleigh_phase(cos_theta);
    float const phase_m = sky_mie_phase(cos_theta);

    float3 optical_depth = (float3)0.0;
    float3 rayleigh = (float3)0.0;
    float3 mie = (float3)0.0;
    float  previous_ray_t = 0.0;

    int const step_count = 64;
    for (int i = 0; i < step_count; ++i)
    {
        float const ray_t =
            pow((float)i / (float)step_count, sample_distribution_exponent) * final_ray_length;
        float const step_size = ray_t - previous_ray_t;
        previous_ray_t = ray_t;

        float3 const p = current_ray_start + ray_dir * ray_t;
        float3 const density = sky_atmosphere_density(sky_atmosphere_height(p));
        optical_depth += density * step_size;

        float3 const view_transmittance = sky_transmittance(optical_depth);
        float3 const light_transmittance =
            sky_transmittance(sky_integrate_optical_depth(p, light_dir));
        float3 const transmittance = view_transmittance * light_transmittance;
        rayleigh += transmittance * phase_r * density.x * step_size;
        mie += transmittance * phase_m * density.y * step_size;
    }

    return (rayleigh * VX_SKY_RAYLEIGH_S + mie * VX_SKY_MIE_S) * light_color;
}

// Longitude is linear in u. Latitude is quadratic in v around v=0.5, allocating more texels to
// directions near the horizon.
float3 sky_view_uv_to_ray_dir(float2 const uv)
{
    float const longitude = uv.x * 2.0 * VX_SKY_PI;
    float const centered_v = uv.y - 0.5;
    float const latitude_sign = centered_v >= 0.0 ? 1.0 : -1.0;
    float const latitude_root = 2.0 * abs(centered_v);
    float const latitude = latitude_sign * latitude_root * latitude_root * VX_SKY_PI * 0.5;
    float const cos_latitude = cos(latitude);
    return float3(cos_latitude * sin(longitude), sin(latitude), cos_latitude * cos(longitude));
}

#endif
