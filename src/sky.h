#pragma once

#include "hlsl_shim.h"

#define VX_SKY_LUT_WIDTH 400
#define VX_SKY_LUT_HEIGHT 400
#define VX_SKY_SUN_COLOR float4(100.f, 100.f, 100.f, 0.f)
#define VX_SKY_SOLAR_RADIUS_RAD (1.5f * 0.0044506f)

typedef struct sky_view_uniforms
{
    float4 view_position;
    float4 sun_direction;
    float4 sun_color;
} sky_view_uniforms;

typedef struct sky_radiance_and_transmittance
{
    float3 radiance;
    float3 transmittance;
} sky_radiance_and_transmittance;

#define VX_SKY_PLANET_RADIUS 6371e3f
#define VX_SKY_PLANET_CENTER float3(0.f, -VX_SKY_PLANET_RADIUS, 0.f)
#define VX_SKY_HEIGHT 100e3f
#define VX_SKY_RAYLEIGH_HEIGHT (VX_SKY_HEIGHT * 0.08f)
#define VX_SKY_MIE_HEIGHT (VX_SKY_HEIGHT * 0.012f)
#define VX_SKY_RAYLEIGH_S float3(5.802e-6f, 13.558e-6f, 33.100e-6f)
#define VX_SKY_MIE_S float3(3.996e-6f, 3.996e-6f, 3.996e-6f)
#define VX_SKY_MIE_A float3(4.40e-6f, 4.40e-6f, 4.40e-6f)
#define VX_SKY_OZONE_A float3(0.650e-6f, 1.881e-6f, 0.085e-6f)

#if defined(__HLSL_VERSION)

#define VX_SKY_INLINE
#define sky_float3_add(a, b) ((a) + (b))
#define sky_float3_mul(a, b) ((a) * (b))
#define sky_float3_scale(a, s) ((a) * (s))
#define sky_float3_sub(a, b) ((a) - (b))
#define sky_float3_dot(a, b) dot((a), (b))
#define sky_float3_length(a) length(a)
#define sky_float3_exp_neg(a) exp(-(a))
#define sky_exp_neg(a) exp(-(a))
#define sky_sqrt(a) sqrt(a)
#define sky_max(a, b) max((a), (b))
#define sky_abs(a) abs(a)

#else

#include <math.h>

#define VX_SKY_INLINE static inline

static inline float3 sky_float3_add(float3 const a, float3 const b)
{
    return float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

static inline float3 sky_float3_mul(float3 const a, float3 const b)
{
    return float3(a.x * b.x, a.y * b.y, a.z * b.z);
}

static inline float3 sky_float3_scale(float3 const a, float const s)
{
    return float3(a.x * s, a.y * s, a.z * s);
}

static inline float3 sky_float3_sub(float3 const a, float3 const b)
{
    return float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

static inline float sky_float3_dot(float3 const a, float3 const b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

static inline float sky_float3_length(float3 const a) { return sqrtf(sky_float3_dot(a, a)); }

static inline float3 sky_float3_exp_neg(float3 const a)
{
    return float3(expf(-a.x), expf(-a.y), expf(-a.z));
}

static inline float sky_exp_neg(float const a) { return expf(-a); }
static inline float sky_sqrt(float const a) { return sqrtf(a); }
static inline float sky_max(float const a, float const b) { return fmaxf(a, b); }
static inline float sky_abs(float const a) { return fabsf(a); }

#endif

VX_SKY_INLINE float2 sky_sphere_intersection(float3 const ray_start, float3 const ray_dir,
                                             float3 const sphere_center, float const sphere_radius)
{
    float3 const oc = sky_float3_sub(ray_start, sphere_center);
    float const  a = sky_float3_dot(ray_dir, oc);
    float const  b = sky_float3_dot(oc, oc) - sphere_radius * sphere_radius;
    float const  discriminant = a * a - b;
    if (discriminant > 0.f)
    {
        float const x = sky_sqrt(discriminant);
        return float2(-a - x, -a + x);
    }
    return float2(-1.f, -1.f);
}

VX_SKY_INLINE float2 sky_atmosphere_intersection(float3 const ray_start, float3 const ray_dir)
{
    return sky_sphere_intersection(ray_start, ray_dir, VX_SKY_PLANET_CENTER,
                                   VX_SKY_PLANET_RADIUS + VX_SKY_HEIGHT);
}

VX_SKY_INLINE float2 sky_planet_intersection(float3 const ray_start, float3 const ray_dir)
{
    return sky_sphere_intersection(ray_start, ray_dir, VX_SKY_PLANET_CENTER, VX_SKY_PLANET_RADIUS);
}

VX_SKY_INLINE float sky_atmosphere_height(float3 const position)
{
    return sky_float3_length(sky_float3_sub(position, VX_SKY_PLANET_CENTER)) - VX_SKY_PLANET_RADIUS;
}

VX_SKY_INLINE float3 sky_atmosphere_density(float const height)
{
    float const rayleigh = sky_exp_neg(sky_max(0.f, height / VX_SKY_RAYLEIGH_HEIGHT));
    float const mie = sky_exp_neg(sky_max(0.f, height / VX_SKY_MIE_HEIGHT));
    float const ozone = sky_max(0.f, 1.f - sky_abs(height - 25000.f) / 15000.f);
    return float3(rayleigh, mie, ozone);
}

VX_SKY_INLINE float3 sky_integrate_optical_depth(float3 const ray_start, float3 const ray_dir)
{
    float2 const intersection = sky_atmosphere_intersection(ray_start, ray_dir);
    int const    sample_count = 8;
    float const  step_size = intersection.y / (float)sample_count;

    float3 optical_depth = float3(0.f, 0.f, 0.f);
    for (int i = 0; i < sample_count; ++i)
    {
        float3 const p =
            sky_float3_add(ray_start, sky_float3_scale(ray_dir, ((float)i + 0.5f) * step_size));
        optical_depth = sky_float3_add(
            optical_depth,
            sky_float3_scale(sky_atmosphere_density(sky_atmosphere_height(p)), step_size));
    }
    return optical_depth;
}

VX_SKY_INLINE float3 sky_transmittance(float3 const optical_depth)
{
    float3 const rayleigh = sky_float3_scale(VX_SKY_RAYLEIGH_S, optical_depth.x);
    float3 const mie =
        sky_float3_scale(sky_float3_add(VX_SKY_MIE_S, VX_SKY_MIE_A), optical_depth.y);
    float3 const ozone = sky_float3_scale(VX_SKY_OZONE_A, optical_depth.z);
    return sky_float3_exp_neg(sky_float3_add(rayleigh, sky_float3_add(mie, ozone)));
}

VX_SKY_INLINE float3 sky_transmitted_sun_color(float3 const view_position,
                                               float3 const sun_direction, float3 const sun_color)
{
    float2 const planet_hit = sky_planet_intersection(view_position, sun_direction);
    if (planet_hit.x >= 0.f)
    {
        return float3(0.f, 0.f, 0.f);
    }
    return sky_float3_mul(
        sun_color, sky_transmittance(sky_integrate_optical_depth(view_position, sun_direction)));
}

#if defined(__HLSL_VERSION)

#define VX_SKY_PI 3.14159265358979323846

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

sky_radiance_and_transmittance
sky_integrate_scattering(float3 const ray_start, float3 const ray_dir, float const ray_length,
                         float3 const light_dir, float3 const light_color)
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

    sky_radiance_and_transmittance result;
    result.radiance = (rayleigh * VX_SKY_RAYLEIGH_S + mie * VX_SKY_MIE_S) * light_color;
    result.transmittance = sky_transmittance(optical_depth);
    return result;
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

float2 sky_view_ray_dir_to_uv(float3 const ray_dir)
{
    float const phi = atan2(ray_dir.x, ray_dir.z);
    float const longitude = phi < 0.0 ? phi + 2.0 * VX_SKY_PI : phi;
    float const latitude = asin(clamp(ray_dir.y, -1.0, 1.0));
    float const latitude_sign = latitude >= 0.0 ? 1.0 : -1.0;
    return float2(longitude / (2.0 * VX_SKY_PI),
                  0.5 + 0.5 * latitude_sign * sqrt(2.0 * abs(latitude) / VX_SKY_PI));
}

#endif
