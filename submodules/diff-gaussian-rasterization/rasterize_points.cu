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

#include "cuda_rasterizer/adam.h"
#include "cuda_rasterizer/config.h"
#include "cuda_rasterizer/rasterizer.h"
#include <cstdio>
#include <cuda_runtime_api.h>
#include <fstream>
#include <functional>
#include <iostream>
#include <math.h>
#include <memory>
#include <sstream>
#include <stdio.h>
#include <string>
#include <torch/extension.h>
#include <tuple>

std::function<char *(size_t N)> resizeFunctional(torch::Tensor &t) {
  auto lambda = [&t](size_t N) {
    t.resize_({(long long)N});
    return reinterpret_cast<char *>(t.contiguous().data_ptr());
  };
  return lambda;
}

std::tuple<int, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor>
RasterizeGaussiansCUDA(
    const torch::Tensor &background, const torch::Tensor &means3D,
    const torch::Tensor &colors, const torch::Tensor &opacity,
    const torch::Tensor &scales, const torch::Tensor &rotations,
    const float scale_modifier, const torch::Tensor &cov3D_precomp,
    const torch::Tensor &viewmatrix, const torch::Tensor &projmatrix,
    const float tan_fovx, const float tan_fovy, const int image_height,
    const int image_width, const torch::Tensor &sh, const int degree,
    const torch::Tensor &campos, const bool prefiltered, const bool debug) {
  if (means3D.ndimension() != 2 || means3D.size(1) != 3) {
    AT_ERROR("means3D must have dimensions (num_points, 3)");
  }

  const int P = means3D.size(0);
  const int H = image_height;
  const int W = image_width;

  auto int_opts = means3D.options().dtype(torch::kInt32);
  auto float_opts = means3D.options().dtype(torch::kFloat32);

  torch::Tensor out_color = torch::full({NUM_CHANNELS, H, W}, 0.0, float_opts);
  torch::Tensor radii =
      torch::full({P}, 0, means3D.options().dtype(torch::kInt32));

  torch::Device device(torch::kCUDA);
  torch::TensorOptions options(torch::kByte);
  torch::Tensor geomBuffer = torch::empty({0}, options.device(device));
  torch::Tensor binningBuffer = torch::empty({0}, options.device(device));
  torch::Tensor imgBuffer = torch::empty({0}, options.device(device));
  std::function<char *(size_t)> geomFunc = resizeFunctional(geomBuffer);
  std::function<char *(size_t)> binningFunc = resizeFunctional(binningBuffer);
  std::function<char *(size_t)> imgFunc = resizeFunctional(imgBuffer);

  int rendered = 0;
  if (P != 0) {
    int M = 0;
    if (sh.size(0) != 0) {
      M = sh.size(1);
    }

    rendered = CudaRasterizer::Rasterizer::forward(
        geomFunc, binningFunc, imgFunc, P, degree, M,
        background.contiguous().data<float>(), W, H,
        means3D.contiguous().data<float>(), sh.contiguous().data_ptr<float>(),
        colors.contiguous().data<float>(), opacity.contiguous().data<float>(),
        scales.contiguous().data_ptr<float>(), scale_modifier,
        rotations.contiguous().data_ptr<float>(),
        cov3D_precomp.contiguous().data<float>(),
        viewmatrix.contiguous().data<float>(),
        projmatrix.contiguous().data<float>(),
        campos.contiguous().data<float>(), tan_fovx, tan_fovy, prefiltered,
        out_color.contiguous().data<float>(), radii.contiguous().data<int>(),
        debug);
  }
  return std::make_tuple(rendered, out_color, radii, geomBuffer, binningBuffer,
                         imgBuffer);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
RasterizeGaussiansBackwardCUDA(
    const torch::Tensor &background, const torch::Tensor &means3D,
    const torch::Tensor &radii, const torch::Tensor &colors,
    const torch::Tensor &scales, const torch::Tensor &rotations,
    const float scale_modifier, const torch::Tensor &cov3D_precomp,
    const torch::Tensor &viewmatrix, const torch::Tensor &projmatrix,
    const float tan_fovx, const float tan_fovy,
    const torch::Tensor &dL_dout_color, const torch::Tensor &sh,
    const int degree, const torch::Tensor &campos,
    const torch::Tensor &geomBuffer, const int R,
    const torch::Tensor &binningBuffer, const torch::Tensor &imageBuffer,
    const bool debug) {
  const int P = means3D.size(0);
  const int H = dL_dout_color.size(1);
  const int W = dL_dout_color.size(2);

  int M = 0;
  if (sh.size(0) != 0) {
    M = sh.size(1);
  }

  torch::Tensor dL_dmeans3D = torch::zeros({P, 3}, means3D.options());
  torch::Tensor dL_dmeans2D = torch::zeros({P, 3}, means3D.options());
  torch::Tensor dL_dcolors = torch::zeros({P, NUM_CHANNELS}, means3D.options());
  torch::Tensor dL_dconic = torch::zeros({P, 2, 2}, means3D.options());
  torch::Tensor dL_dopacity = torch::zeros({P, 1}, means3D.options());
  torch::Tensor dL_dcov3D = torch::zeros({P, 6}, means3D.options());
  torch::Tensor dL_dsh = torch::zeros({P, M, 3}, means3D.options());
  torch::Tensor dL_dscales = torch::zeros({P, 3}, means3D.options());
  torch::Tensor dL_drotations = torch::zeros({P, 4}, means3D.options());

  if (P != 0) {
    CudaRasterizer::Rasterizer::backward(
        P, degree, M, R, background.contiguous().data<float>(), W, H,
        means3D.contiguous().data<float>(), sh.contiguous().data<float>(),
        colors.contiguous().data<float>(), scales.data_ptr<float>(),
        scale_modifier, rotations.data_ptr<float>(),
        cov3D_precomp.contiguous().data<float>(),
        viewmatrix.contiguous().data<float>(),
        projmatrix.contiguous().data<float>(),
        campos.contiguous().data<float>(), tan_fovx, tan_fovy,
        radii.contiguous().data<int>(),
        reinterpret_cast<char *>(geomBuffer.contiguous().data_ptr()),
        reinterpret_cast<char *>(binningBuffer.contiguous().data_ptr()),
        reinterpret_cast<char *>(imageBuffer.contiguous().data_ptr()),
        dL_dout_color.contiguous().data<float>(),
        dL_dmeans2D.contiguous().data<float>(),
        dL_dconic.contiguous().data<float>(),
        dL_dopacity.contiguous().data<float>(),
        dL_dcolors.contiguous().data<float>(),
        dL_dmeans3D.contiguous().data<float>(),
        dL_dcov3D.contiguous().data<float>(), dL_dsh.contiguous().data<float>(),
        dL_dscales.contiguous().data<float>(),
        dL_drotations.contiguous().data<float>(), debug);
  }

  return std::make_tuple(dL_dmeans2D, dL_dcolors, dL_dopacity, dL_dmeans3D,
                         dL_dcov3D, dL_dsh, dL_dscales, dL_drotations);
}

torch::Tensor markVisible(torch::Tensor &means3D, torch::Tensor &viewmatrix,
                          torch::Tensor &projmatrix) {
  const int P = means3D.size(0);

  torch::Tensor present =
      torch::full({P}, false, means3D.options().dtype(at::kBool));

  if (P != 0) {
    CudaRasterizer::Rasterizer::markVisible(
        P, means3D.contiguous().data<float>(),
        viewmatrix.contiguous().data<float>(),
        projmatrix.contiguous().data<float>(),
        present.contiguous().data<bool>());
  }

  return present;
}

std::tuple<int, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
RasterizeGaussianImportancesCUDA(
    const torch::Tensor &means3D, const torch::Tensor &opacity,
    const torch::Tensor &scales, const torch::Tensor &rotations,
    const float scale_modifier, const torch::Tensor &cov3D_precomp,
    const torch::Tensor &viewmatrix, const torch::Tensor &projmatrix,
    const torch::Tensor &weightmap, const float tan_fovx, const float tan_fovy,
    const int image_height, const int image_width, const bool prefiltered,
    const bool debug) {
  if (means3D.ndimension() != 2 || means3D.size(1) != 3) {
    AT_ERROR("means3D must have dimensions (num_points, 3)");
  }

  const int P = means3D.size(0);
  const int H = image_height;
  const int W = image_width;

  auto int_opts = means3D.options().dtype(torch::kInt32);
  auto float_opts = means3D.options().dtype(torch::kFloat32);

  //   torch::Tensor out_color = torch::full({NUM_CHANNELS, H, W}, 0.0,
  //   float_opts);
  torch::Tensor radii =
      torch::full({P}, 0, means3D.options().dtype(torch::kInt32));

  torch::Tensor accum_weights = torch::full({P}, 0, float_opts);
  torch::Tensor accum_scaled_weights = torch::full({P}, 0, float_opts);
  torch::Tensor num_hit_pixels =
      torch::full({P}, 0, int_opts); // Number of pixels hit by each Gaussian
  torch::Tensor num_max_pixels =
      torch::full({P}, 0, int_opts); // Number of pixels where each Gaussian had
                                     // max blending weight (alpha * T)

  torch::Device device(torch::kCUDA);
  torch::TensorOptions options(torch::kByte);
  torch::Tensor geomBuffer = torch::empty({0}, options.device(device));
  torch::Tensor binningBuffer = torch::empty({0}, options.device(device));
  torch::Tensor imgBuffer = torch::empty({0}, options.device(device));
  std::function<char *(size_t)> geomFunc = resizeFunctional(geomBuffer);
  std::function<char *(size_t)> binningFunc = resizeFunctional(binningBuffer);
  std::function<char *(size_t)> imgFunc = resizeFunctional(imgBuffer);

  int rendered = 0;
  if (P != 0) {
    int M = 0;
    torch::Tensor bg = torch::full({NUM_CHANNELS}, 0, float_opts);
    torch::Tensor color = torch::full({P, NUM_CHANNELS}, 0.0, float_opts);
    torch::Tensor campos = torch::full({3}, 0, float_opts);

    const float *wptr = nullptr;
    if (weightmap.defined() && weightmap.numel() > 0) {
      if (weightmap.size(0) == H && weightmap.size(1) == W) {
        wptr = weightmap.contiguous().data<float>();
      }
    }

    rendered = CudaRasterizer::Rasterizer::rasterize_importance(
        geomFunc, binningFunc, imgFunc, P, 0, M,
        bg.contiguous().data_ptr<float>(), W, H,
        means3D.contiguous().data<float>(), nullptr,
        color.contiguous().data<float>(), opacity.contiguous().data<float>(),
        scales.contiguous().data_ptr<float>(), scale_modifier,
        rotations.contiguous().data_ptr<float>(),
        cov3D_precomp.contiguous().data<float>(),
        viewmatrix.contiguous().data<float>(),
        projmatrix.contiguous().data<float>(),
        campos.contiguous().data<float>(), tan_fovx, tan_fovy, prefiltered,
        wptr, radii.contiguous().data<int>(),
        accum_weights.contiguous().data<float>(),
        accum_scaled_weights.contiguous().data<float>(),
        num_hit_pixels.contiguous().data<int>(),
        num_max_pixels.contiguous().data<int>(), debug);
  }
  return std::make_tuple(rendered, radii, geomBuffer, binningBuffer, imgBuffer,
                         accum_weights, accum_scaled_weights, num_hit_pixels,
                         num_max_pixels);
}

std::tuple<int, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor>
RasterizeGaussiansWithZBufCUDA(
    const torch::Tensor &background, const torch::Tensor &means3D,
    const torch::Tensor &colors, const torch::Tensor &opacity,
    const torch::Tensor &scales, const torch::Tensor &rotations,
    const float scale_modifier, const torch::Tensor &cov3D_precomp,
    const torch::Tensor &viewmatrix, const torch::Tensor &projmatrix,
    const float tan_fovx, const float tan_fovy, const int image_height,
    const int image_width, const torch::Tensor &sh, const int degree,
    const torch::Tensor &campos, const torch::Tensor &zbuf,
    const float tolerance, const bool prefiltered, const bool debug) {
  if (means3D.ndimension() != 2 || means3D.size(1) != 3) {
    AT_ERROR("means3D must have dimensions (num_points, 3)");
  }

  const int P = means3D.size(0);
  const int H = image_height;
  const int W = image_width;

  auto int_opts = means3D.options().dtype(torch::kInt32);
  auto float_opts = means3D.options().dtype(torch::kFloat32);

  torch::Tensor out_color = torch::full({NUM_CHANNELS, H, W}, 0.0, float_opts);
  torch::Tensor radii =
      torch::full({P}, 0, means3D.options().dtype(torch::kInt32));

  torch::Device device(torch::kCUDA);
  torch::TensorOptions options(torch::kByte);
  torch::Tensor geomBuffer = torch::empty({0}, options.device(device));
  torch::Tensor binningBuffer = torch::empty({0}, options.device(device));
  torch::Tensor imgBuffer = torch::empty({0}, options.device(device));
  std::function<char *(size_t)> geomFunc = resizeFunctional(geomBuffer);
  std::function<char *(size_t)> binningFunc = resizeFunctional(binningBuffer);
  std::function<char *(size_t)> imgFunc = resizeFunctional(imgBuffer);

  int rendered = 0;
  if (P != 0) {
    int M = 0;
    if (sh.size(0) != 0) {
      M = sh.size(1);
    }

    const float *zptr = nullptr;
    if (zbuf.defined() && zbuf.numel() > 0) {
      zptr = zbuf.contiguous().data<float>();
    }

    rendered = CudaRasterizer::Rasterizer::rasterize_with_zbuf(
        geomFunc, binningFunc, imgFunc, P, degree, M,
        background.contiguous().data<float>(), W, H,
        means3D.contiguous().data<float>(), sh.contiguous().data_ptr<float>(),
        colors.contiguous().data<float>(), opacity.contiguous().data<float>(),
        scales.contiguous().data_ptr<float>(), scale_modifier,
        rotations.contiguous().data_ptr<float>(),
        cov3D_precomp.contiguous().data<float>(),
        viewmatrix.contiguous().data<float>(),
        projmatrix.contiguous().data<float>(),
        campos.contiguous().data<float>(), tan_fovx, tan_fovy, prefiltered,
        out_color.contiguous().data<float>(), radii.contiguous().data<int>(),
        zptr, tolerance, debug);
  }
  return std::make_tuple(rendered, out_color, radii, geomBuffer, binningBuffer,
                         imgBuffer);
}

void adamUpdate(torch::Tensor &param, torch::Tensor &param_grad,
                torch::Tensor &exp_avg, torch::Tensor &exp_avg_sq,
                torch::Tensor &visible, const float lr, const float b1,
                const float b2, const float eps, const uint32_t N,
                const uint32_t M) {
  ADAM::adamUpdate(
      param.contiguous().data<float>(), param_grad.contiguous().data<float>(),
      exp_avg.contiguous().data<float>(), exp_avg_sq.contiguous().data<float>(),
      visible.contiguous().data<bool>(), lr, b1, b2, eps, N, M);
}