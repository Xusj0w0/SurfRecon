/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#ifndef CUDA_RASTERIZER_FORWARD_H_INCLUDED
#define CUDA_RASTERIZER_FORWARD_H_INCLUDED

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

namespace FORWARD {
// Perform initial steps for each Gaussian prior to rasterization.
void preprocess(int P, int D, int M, const float *orig_points,
                const glm::vec3 *scales, const float scale_modifier,
                const glm::vec4 *rotations, const float *opacities,
                const float *shs, bool *clamped, const float *cov3D_precomp,
                const float *colors_precomp, const float *viewmatrix,
                const float *projmatrix, const glm::vec3 *cam_pos, const int W,
                int H, const float focal_x, float focal_y, const float tan_fovx,
                float tan_fovy, int *radii, float2 *points_xy_image,
                float *depths, float *cov3Ds, float *colors,
                float4 *conic_opacity, const dim3 grid, uint32_t *tiles_touched,
                bool prefiltered);

// Main rasterization method.
void render(const dim3 grid, dim3 block, const uint2 *ranges,
            const uint32_t *point_list, int W, int H,
            const float2 *points_xy_image, const float *features,
            const float4 *conic_opacity, float *final_T, uint32_t *n_contrib,
            const float *bg_color, float *out_color);

void rasterize_importance(const dim3 grid, dim3 block, const uint2 *ranges,
                          const uint32_t *point_list, int W, int H,
                          const float2 *points_xy_image,
                          const float4 *conic_opacity, const float *weightmap,
                          float *final_T, uint32_t *n_contrib,
                          float *accum_weights, float *accum_scaled_weights,
                          int *num_hit_pixels, int *num_max_pixels);

void preprocess_with_zbuf(
    int P, int D, int M, const float *means3D, const glm::vec3 *scales,
    const float scale_modifier, const glm::vec4 *rotations,
    const float *opacities, const float *shs, bool *clamped,
    const float *cov3D_precomp, const float *colors_precomp,
    const float *viewmatrix, const float *projmatrix, const glm::vec3 *cam_pos,
    const int W, int H, const float focal_x, float focal_y,
    const float tan_fovx, float tan_fovy, int *radii, float2 *means2D,
    float *depths, float *cov3Ds, float *rgb, float4 *conic_opacity,
    const dim3 grid, uint32_t *tiles_touched, const float *zbuf,
    const float tolerance, bool prefiltered);

} // namespace FORWARD

#endif