// @file roipooling_gpu.cu
// @brief ROI pooling block implementation (GPU)
// Original code : https://github.com/rbgirshick/caffe-fast-rcnn/blob/bcd9b4eadc7d8fbc433aeefd564e82ec63aaf69c/src/caffe/layers/roi_pooling_layer.cu
// Modify for matlab column-based array
//
// @author Tuan-Hung Vu

/*
Copyright (C) 2015-2016 Tuan-Hung Vu.
All rights reserved.

This file is made available under the terms of the BSD license.
*/

#include "roipooling.hpp"
#include "../datacu.hpp"
#include <assert.h>
#include <float.h>
#include <sm_20_atomic_functions.h>
#include <stdio.h>

/* ---------------------------------------------------------------- */
/*                                           roipooling_max_forward */
/* ---------------------------------------------------------------- */



// from fast-rcnn's cu kernel
template<typename Dtype>
__global__ void roipooling_max_forward_kernel(const int nthreads,
 const Dtype* bottom_data,
 const float spatial_scale,
 const size_t channels,
 const size_t height,
 const size_t width,
 const size_t pooled_height,
 const size_t pooled_width,
 const Dtype* bottom_rois,
 Dtype* top_data,
 Dtype* argmax_data)
{
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index < nthreads) {
  // (n, c, ph, pw) is an element in the pooled output
  int pw = index % pooled_width;
  int ph = (index / pooled_width) % pooled_height;
  int c = (index / pooled_width / pooled_height) % channels;
  int n = index / pooled_width / pooled_height / channels;

  //printf("idx,w,h,c,n = %d %d %d %d %d \n", index, pw, ph, c, n);

  bottom_rois += n * 5;
  int roi_batch_ind = bottom_rois[0];
  int roi_start_w = round(bottom_rois[1] * spatial_scale);
  int roi_start_h = round(bottom_rois[2] * spatial_scale);
  int roi_end_w = round(bottom_rois[3] * spatial_scale);
  int roi_end_h = round(bottom_rois[4] * spatial_scale);
  //printf("rbind, sw, sh, ew, eh = %d %d %d %d %d \n", roi_batch_ind, roi_start_w, roi_start_h, roi_end_w, roi_end_h);

  // Force malformed ROIs to be 1x1
  int roi_width = max(roi_end_w - roi_start_w + 1, 1);
  int roi_height = max(roi_end_h - roi_start_h + 1, 1);
  Dtype bin_size_h = static_cast<Dtype>(roi_height)
                       / static_cast<Dtype>(pooled_height);
  Dtype bin_size_w = static_cast<Dtype>(roi_width)
                       / static_cast<Dtype>(pooled_width);

  int hstart = static_cast<int>(floor(static_cast<Dtype>(ph)
                                        * bin_size_h));
  int wstart = static_cast<int>(floor(static_cast<Dtype>(pw)
                                        * bin_size_w));
  int hend = static_cast<int>(ceil(static_cast<Dtype>(ph + 1)
                                     * bin_size_h));
  int wend = static_cast<int>(ceil(static_cast<Dtype>(pw + 1)
                                     * bin_size_w));

  //printf("%d %d %d %d %d %d %d %d \n", roi_width, roi_height, bin_size_h, bin_size_w, hstart, wstart, hend, wend); 

  // Add roi offsets and clip to input boundaries
  hstart = min(max(hstart + roi_start_h, 0), (int)height);
  hend = min(max(hend + roi_start_h, 0), (int)height);
  wstart = min(max(wstart + roi_start_w, 0), (int)width);
  wend = min(max(wend + roi_start_w, 0), (int)width);
  bool is_empty = (hend <= hstart) || (wend <= wstart);

  //printf("%d %d %d %d \n", hstart, hend, wstart, wend);

  // Define an empty pooling region to be zero
  Dtype maxval = is_empty ? 0 : -FLT_MAX;
  // If nothing is pooled, argmax = -1 causes nothing to be backprop'd
  int maxidx = -1;
  bottom_data += (roi_batch_ind * channels + c) * height * width;
  for (int h = hstart; h < hend; ++h) {
    for (int w = wstart; w < wend; ++w) {
      //int bottom_index = h * width + w;
      int bottom_index = w*height + h;
      if (bottom_data[bottom_index] > maxval) {
        maxval = bottom_data[bottom_index];
        maxidx = bottom_index;
      }
    }
  }
  top_data[index] = maxval;
  argmax_data[index] = maxidx;
  }
}

template<> vl::Error
vl::impl::roipooling_max_forward<vl::GPU, float>(float* pooled,
                                      float* argmax,
                                      float const* data,
                                      float const* rois,
                                      size_t height, size_t width, size_t channels, size_t num_rois,
                                      size_t poolHeight, size_t poolWidth,
                                      float spatial_scale)
{
  int nthreads = poolWidth * poolHeight * channels * num_rois;

  roipooling_max_forward_kernel<float>
  <<< divideUpwards(nthreads, VL_CUDA_NUM_THREADS), VL_CUDA_NUM_THREADS >>>
  (nthreads, data, spatial_scale, channels, height, width, poolHeight, poolWidth,
  rois, pooled, argmax);

  cudaError_t status = cudaPeekAtLastError() ;
  return (status == cudaSuccess) ? vl::vlSuccess : vl::vlErrorCuda ;
}

/* ---------------------------------------------------------------- */
/*                                           roipooling_max_backward */
/* ---------------------------------------------------------------- */
// from fast-rcnn's cu kernel
template <typename Dtype>
__global__ void roipooling_max_backward_kernel(const int nthreads, const Dtype* top_diff,
    const Dtype* argmax_data, const int num_rois, const Dtype spatial_scale,
    const int channels, const int height, const int width,
    const int pooled_height, const int pooled_width, Dtype* bottom_diff,
    const Dtype* bottom_rois) 
{
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index < nthreads) {
  // (n, c, h, w) coords in bottom data
  int h = index % height;
  int w = (index / height) % width;
  int c = (index / width / height) % channels;
  int n = index / width / height / channels;

  Dtype gradient = 0;
  // Accumulate gradient over all ROIs that pooled this element
  for (int roi_n = 0; roi_n < num_rois; ++roi_n) {
    const Dtype* offset_bottom_rois = bottom_rois + roi_n * 5;
    int roi_batch_ind = offset_bottom_rois[0];
    // Skip if ROI's batch index doesn't match n
    if (n != roi_batch_ind) {
      continue;
    }

    int roi_start_w = round(offset_bottom_rois[1] * spatial_scale);
    int roi_start_h = round(offset_bottom_rois[2] * spatial_scale);
    int roi_end_w = round(offset_bottom_rois[3] * spatial_scale);
    int roi_end_h = round(offset_bottom_rois[4] * spatial_scale);

    // Skip if ROI doesn't include (h, w)
    const bool in_roi = (w >= roi_start_w && w <= roi_end_w &&
                         h >= roi_start_h && h <= roi_end_h);
    if (!in_roi) {
      continue;
    }

    int offset = (roi_n * channels + c) * pooled_height * pooled_width;
    const Dtype* offset_top_diff = top_diff + offset;
    const float* offset_argmax_data = argmax_data + offset;

    // Compute feasible set of pooled units that could have pooled
    // this bottom unit

    // Force malformed ROIs to be 1x1
    int roi_width = max(roi_end_w - roi_start_w + 1, 1);
    int roi_height = max(roi_end_h - roi_start_h + 1, 1);

    Dtype bin_size_h = static_cast<Dtype>(roi_height)
                         / static_cast<Dtype>(pooled_height);
    Dtype bin_size_w = static_cast<Dtype>(roi_width)
                       / static_cast<Dtype>(pooled_width);

    int phstart = floor(static_cast<Dtype>(h - roi_start_h) / bin_size_h);
    int phend = ceil(static_cast<Dtype>(h - roi_start_h + 1) / bin_size_h);
    int pwstart = floor(static_cast<Dtype>(w - roi_start_w) / bin_size_w);
    int pwend = ceil(static_cast<Dtype>(w - roi_start_w + 1) / bin_size_w);

    phstart = min(max(phstart, 0), pooled_height);
    phend = min(max(phend, 0), pooled_height);
    pwstart = min(max(pwstart, 0), pooled_width);
    pwend = min(max(pwend, 0), pooled_width);

    for (int ph = phstart; ph < phend; ++ph) {
      for (int pw = pwstart; pw < pwend; ++pw) {
        //if (offset_argmax_data[ph * pooled_width + pw] == (h * width + w)) {
        if (offset_argmax_data[ph * pooled_width + pw] == w*height + h) {
          gradient += offset_top_diff[ph * pooled_width + pw];
        }
      }
    }
  }
  bottom_diff[index] = gradient;
  }
}

template<> vl::Error
vl::impl::roipooling_max_backward<vl::GPU, float>(float* derData,
                                       float const* data,
                                       float const* rois,
                                       float const* derPooled,
                                       float const* argmax,
                                       size_t height, size_t width, size_t channels, size_t num_data, size_t num_rois,
                                       size_t poolHeight, size_t poolWidth,
                                       float spatial_scale)
{
  int nthreads = width * height * channels * num_data;

  roipooling_max_backward_kernel<float>
  <<< divideUpwards(nthreads, VL_CUDA_NUM_THREADS), VL_CUDA_NUM_THREADS >>>
      (nthreads, derPooled, argmax, num_rois, spatial_scale, channels, height, width,
      poolHeight, poolWidth, derData, rois);

  cudaError_t status = cudaPeekAtLastError() ;
  return (status == cudaSuccess) ? vl::vlSuccess : vl::vlErrorCuda ;
}
