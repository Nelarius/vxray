#include "cvox.h"

#define OGT_VOX_IMPLEMENTATION
#include <ogt_vox.h>

#include <stddef.h>
#include <stdint.h>

static_assert(sizeof(cvox_rgba) == sizeof(ogt_vox_rgba));
static_assert(sizeof(cvox_transform) == sizeof(ogt_vox_transform));
static_assert(sizeof(cvox_palette) == sizeof(ogt_vox_palette));
static_assert(sizeof(cvox_matl) == sizeof(ogt_vox_matl));
static_assert(sizeof(cvox_matl_array) == sizeof(ogt_vox_matl_array));
static_assert(sizeof(cvox_cam) == sizeof(ogt_vox_cam));
static_assert(sizeof(cvox_sun) == sizeof(ogt_vox_sun));
static_assert(sizeof(cvox_model) == sizeof(ogt_vox_model));
static_assert(sizeof(cvox_keyframe_transform) == sizeof(ogt_vox_keyframe_transform));
static_assert(sizeof(cvox_keyframe_model) == sizeof(ogt_vox_keyframe_model));
static_assert(sizeof(cvox_anim_transform) == sizeof(ogt_vox_anim_transform));
static_assert(sizeof(cvox_anim_model) == sizeof(ogt_vox_anim_model));
static_assert(sizeof(cvox_instance) == sizeof(ogt_vox_instance));
static_assert(sizeof(cvox_layer) == sizeof(ogt_vox_layer));
static_assert(sizeof(cvox_group) == sizeof(ogt_vox_group));
static_assert(sizeof(cvox_scene) == sizeof(ogt_vox_scene));

static_assert(alignof(cvox_rgba) == alignof(ogt_vox_rgba));
static_assert(alignof(cvox_transform) == alignof(ogt_vox_transform));
static_assert(alignof(cvox_scene) == alignof(ogt_vox_scene));

extern "C" {

void cvox_set_memory_allocator(cvox_alloc_func const alloc_func, cvox_free_func const free_func)
{
    ogt_vox_set_memory_allocator(
        reinterpret_cast<ogt_vox_alloc_func>(alloc_func),
        reinterpret_cast<ogt_vox_free_func>(free_func));
}

void* cvox_malloc(size_t const size) { return ogt_vox_malloc(size); }

void cvox_free(void* const mem) { ogt_vox_free(mem); }

void cvox_set_progress_callback_func(
    cvox_progress_callback_func const progress_callback_func, void* const user_data)
{
    ogt_vox_set_progress_callback_func(
        reinterpret_cast<ogt_vox_progress_callback_func>(progress_callback_func), user_data);
}

cvox_transform cvox_transform_get_identity(void)
{
    ogt_vox_transform const transform = ogt_vox_transform_get_identity();
    return *reinterpret_cast<cvox_transform const*>(&transform);
}

cvox_transform cvox_transform_multiply(cvox_transform const* const a, cvox_transform const* const b)
{
    ogt_vox_transform const transform = ogt_vox_transform_multiply(
        *reinterpret_cast<ogt_vox_transform const*>(a),
        *reinterpret_cast<ogt_vox_transform const*>(b));
    return *reinterpret_cast<cvox_transform const*>(&transform);
}

cvox_scene const* cvox_read_scene(uint8_t const* const buffer, uint32_t const buffer_size)
{
    return reinterpret_cast<cvox_scene const*>(ogt_vox_read_scene(buffer, buffer_size));
}

cvox_scene const* cvox_read_scene_with_flags(
    uint8_t const* const buffer, uint32_t const buffer_size, uint32_t const read_flags)
{
    return reinterpret_cast<cvox_scene const*>(
        ogt_vox_read_scene_with_flags(buffer, buffer_size, read_flags));
}

void cvox_destroy_scene(cvox_scene const* const scene)
{
    ogt_vox_destroy_scene(reinterpret_cast<ogt_vox_scene const*>(scene));
}

uint8_t* cvox_write_scene(cvox_scene const* const scene, uint32_t* const buffer_size)
{
    return ogt_vox_write_scene(reinterpret_cast<ogt_vox_scene const*>(scene), buffer_size);
}

void cvox_camera_to_transform(cvox_cam const* const camera, cvox_transform* const transform)
{
    ogt_vox_camera_to_transform(
        reinterpret_cast<ogt_vox_cam const*>(camera),
        reinterpret_cast<ogt_vox_transform*>(transform));
}

cvox_scene* cvox_merge_scenes(
    cvox_scene const** const scenes, uint32_t const scene_count,
    cvox_rgba const* const required_colors, uint32_t const required_color_count)
{
    return reinterpret_cast<cvox_scene*>(ogt_vox_merge_scenes(
        reinterpret_cast<ogt_vox_scene const**>(scenes), scene_count,
        reinterpret_cast<ogt_vox_rgba const*>(required_colors), required_color_count));
}

uint32_t cvox_sample_instance_model(cvox_instance const* const instance, uint32_t const frame_index)
{
    return ogt_vox_sample_instance_model(
        reinterpret_cast<ogt_vox_instance const*>(instance), frame_index);
}

cvox_transform cvox_sample_instance_transform_global(
    cvox_instance const* const instance, uint32_t const frame_index, cvox_scene const* const scene)
{
    ogt_vox_transform const transform = ogt_vox_sample_instance_transform_global(
        reinterpret_cast<ogt_vox_instance const*>(instance), frame_index,
        reinterpret_cast<ogt_vox_scene const*>(scene));
    return *reinterpret_cast<cvox_transform const*>(&transform);
}

cvox_transform cvox_sample_instance_transform_local(
    cvox_instance const* const instance, uint32_t const frame_index)
{
    ogt_vox_transform const transform = ogt_vox_sample_instance_transform_local(
        reinterpret_cast<ogt_vox_instance const*>(instance), frame_index);
    return *reinterpret_cast<cvox_transform const*>(&transform);
}

cvox_transform cvox_sample_group_transform_global(
    cvox_group const* const group, uint32_t const frame_index, cvox_scene const* const scene)
{
    ogt_vox_transform const transform = ogt_vox_sample_group_transform_global(
        reinterpret_cast<ogt_vox_group const*>(group), frame_index,
        reinterpret_cast<ogt_vox_scene const*>(scene));
    return *reinterpret_cast<cvox_transform const*>(&transform);
}

cvox_transform
cvox_sample_group_transform_local(cvox_group const* const group, uint32_t const frame_index)
{
    ogt_vox_transform const transform = ogt_vox_sample_group_transform_local(
        reinterpret_cast<ogt_vox_group const*>(group), frame_index);
    return *reinterpret_cast<cvox_transform const*>(&transform);
}
}
