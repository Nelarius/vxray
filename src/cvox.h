#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

// Refer to wrapped ogt_vox.h for short manual.

#ifdef __cplusplus
extern "C" {
#endif

enum
{
    CVOX_INVALID_GROUP_INDEX = UINT32_MAX,
    CVOX_INVALID_LAYER_INDEX = UINT32_MAX,
};

typedef struct cvox_rgba
{
    uint8_t r;
    uint8_t g;
    uint8_t b;
    uint8_t a;
} cvox_rgba;

typedef struct cvox_transform
{
    float m00, m01, m02, m03;
    float m10, m11, m12, m13;
    float m20, m21, m22, m23;
    float m30, m31, m32, m33;
} cvox_transform;

typedef struct cvox_palette
{
    cvox_rgba color[256];
} cvox_palette;

typedef enum cvox_matl_type
{
    cvox_matl_type_diffuse = 0,
    cvox_matl_type_metal = 1,
    cvox_matl_type_glass = 2,
    cvox_matl_type_emit = 3,
    cvox_matl_type_blend = 4,
    cvox_matl_type_media = 5,
} cvox_matl_type;

typedef enum cvox_cam_mode
{
    cvox_cam_mode_perspective = 0,
    cvox_cam_mode_free = 1,
    cvox_cam_mode_pano = 2,
    cvox_cam_mode_orthographic = 3,
    cvox_cam_mode_isometric = 4,
    cvox_cam_mode_unknown = 5
} cvox_cam_mode;

enum
{
    CVOX_MATL_HAVE_METAL = 1u << 0,
    CVOX_MATL_HAVE_ROUGH = 1u << 1,
    CVOX_MATL_HAVE_SPEC = 1u << 2,
    CVOX_MATL_HAVE_IOR = 1u << 3,
    CVOX_MATL_HAVE_ATT = 1u << 4,
    CVOX_MATL_HAVE_FLUX = 1u << 5,
    CVOX_MATL_HAVE_EMIT = 1u << 6,
    CVOX_MATL_HAVE_LDR = 1u << 7,
    CVOX_MATL_HAVE_TRANS = 1u << 8,
    CVOX_MATL_HAVE_ALPHA = 1u << 9,
    CVOX_MATL_HAVE_D = 1u << 10,
    CVOX_MATL_HAVE_SP = 1u << 11,
    CVOX_MATL_HAVE_G = 1u << 12,
    CVOX_MATL_HAVE_MEDIA = 1u << 13,
};

typedef enum cvox_media_type
{
    cvox_media_type_absorb,
    cvox_media_type_scatter,
    cvox_media_type_emit,
    cvox_media_type_sss,
} cvox_media_type;

typedef struct cvox_matl
{
    uint32_t        content_flags;
    cvox_media_type media_type;
    cvox_matl_type  type;
    float           metal;
    float           rough;
    float           spec;
    float           ior;
    float           att;
    float           flux;
    float           emit;
    float           ldr;
    float           trans;
    float           alpha;
    float           d;
    float           sp;
    float           g;
    float           media;
} cvox_matl;

typedef struct cvox_matl_array
{
    cvox_matl matl[256];
} cvox_matl_array;

typedef struct cvox_cam
{
    uint32_t      camera_id;
    cvox_cam_mode mode;
    float         focus[3];
    float         angle[3];
    float         radius;
    float         frustum;
    int           fov;
} cvox_cam;

typedef struct cvox_sun
{
    float     intensity;
    float     area;
    float     angle[2];
    cvox_rgba rgba;
    bool      disk;
} cvox_sun;

typedef struct cvox_model
{
    uint32_t       size_x;
    uint32_t       size_y;
    uint32_t       size_z;
    uint32_t       voxel_hash;
    uint8_t const* voxel_data;
} cvox_model;

typedef struct cvox_keyframe_transform
{
    uint32_t       frame_index;
    cvox_transform transform;
} cvox_keyframe_transform;

typedef struct cvox_keyframe_model
{
    uint32_t frame_index;
    uint32_t model_index;
} cvox_keyframe_model;

typedef struct cvox_anim_transform
{
    cvox_keyframe_transform const* keyframes;
    uint32_t                       num_keyframes;
    bool                           loop;
} cvox_anim_transform;

typedef struct cvox_anim_model
{
    cvox_keyframe_model const* keyframes;
    uint32_t                   num_keyframes;
    bool                       loop;
} cvox_anim_model;

typedef struct cvox_instance
{
    char const*         name;
    cvox_transform      transform;
    uint32_t            model_index;
    uint32_t            layer_index;
    uint32_t            group_index;
    bool                hidden;
    cvox_anim_transform transform_anim;
    cvox_anim_model     model_anim;
} cvox_instance;

typedef struct cvox_layer
{
    char const* name;
    cvox_rgba   color;
    bool        hidden;
} cvox_layer;

typedef struct cvox_group
{
    char const*         name;
    cvox_transform      transform;
    uint32_t            parent_group_index;
    uint32_t            layer_index;
    bool                hidden;
    cvox_anim_transform transform_anim;
} cvox_group;

typedef struct cvox_scene
{
    uint32_t             file_version;
    uint32_t             num_models;
    uint32_t             num_instances;
    uint32_t             num_layers;
    uint32_t             num_groups;
    uint32_t             num_color_names;
    char const**         color_names;
    cvox_model const**   models;
    cvox_instance const* instances;
    cvox_layer const*    layers;
    cvox_group const*    groups;
    cvox_palette         palette;
    cvox_matl_array      materials;
    uint32_t             num_cameras;
    cvox_cam const*      cameras;
    cvox_sun*            sun;
    uint32_t             anim_range_start;
    uint32_t             anim_range_end;
} cvox_scene;

typedef void* (*cvox_alloc_func)(size_t size);
typedef void (*cvox_free_func)(void* ptr);
typedef bool (*cvox_progress_callback_func)(float progress, void* user_data);

enum
{
    CVOX_READ_SCENE_FLAGS_GROUPS = 1u << 0,
    CVOX_READ_SCENE_FLAGS_KEYFRAMES = 1u << 1,
    CVOX_READ_SCENE_FLAGS_KEEP_EMPTY_MODELS_INSTANCES = 1u << 2,
    CVOX_READ_SCENE_FLAGS_KEEP_DUPLICATE_MODELS = 1u << 3,
};

void  cvox_set_memory_allocator(cvox_alloc_func alloc_func, cvox_free_func free_func);
void* cvox_malloc(size_t size);
void  cvox_free(void* mem);
void  cvox_set_progress_callback_func(
     cvox_progress_callback_func progress_callback_func, void* user_data);

cvox_transform cvox_transform_get_identity(void);
cvox_transform cvox_transform_multiply(cvox_transform const* a, cvox_transform const* b);

cvox_scene const* cvox_read_scene(uint8_t const* buffer, uint32_t buffer_size);
cvox_scene const*
     cvox_read_scene_with_flags(uint8_t const* buffer, uint32_t buffer_size, uint32_t read_flags);
void cvox_destroy_scene(cvox_scene const* scene);
uint8_t*    cvox_write_scene(cvox_scene const* scene, uint32_t* buffer_size);
void        cvox_camera_to_transform(cvox_cam const* camera, cvox_transform* transform);
cvox_scene* cvox_merge_scenes(
    cvox_scene const** scenes, uint32_t scene_count, cvox_rgba const* required_colors,
    uint32_t required_color_count);
uint32_t       cvox_sample_instance_model(cvox_instance const* instance, uint32_t frame_index);
cvox_transform cvox_sample_instance_transform_global(
    cvox_instance const* instance, uint32_t frame_index, cvox_scene const* scene);
cvox_transform
cvox_sample_instance_transform_local(cvox_instance const* instance, uint32_t frame_index);
cvox_transform cvox_sample_group_transform_global(
    cvox_group const* group, uint32_t frame_index, cvox_scene const* scene);
cvox_transform cvox_sample_group_transform_local(cvox_group const* group, uint32_t frame_index);

#ifdef __cplusplus
}
#endif
