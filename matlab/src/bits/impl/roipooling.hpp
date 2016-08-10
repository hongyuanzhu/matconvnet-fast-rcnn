// @file roipooling.hpp
// @brief ROI pooling block implementation
// @author Tuan-Hung Vu

/*
Copyright (C) 2015-2016 Tuan-Hung Vu.
All rights reserved.

This file is made available under the terms of the BSD license.
*/

#ifndef VL_NNROIPOOLING_H
#define VL_NNROIPOOLING_H

#include "../data.hpp"
#include <cstddef>

namespace vl { namespace impl {

  /* Max pooling */

  template<vl::Device dev, typename type> vl::Error
  roipooling_max_forward(type* pooled,
                      type* argmax,
                      type const* data,
                      type const* rois,
                      size_t height, size_t width, size_t channels,
                      size_t num_rois,
                      size_t poolHeight, size_t poolWidth,
                      float spatial_scale) ;

  template<vl::Device dev, typename type> vl::Error
  roipooling_max_backward(type* derData,
                       type const* data,
                       type const* rois,
                       type const* derPooled,
                       type const* argmax,
                       size_t height, size_t width, size_t channels,
                       size_t num_data, size_t num_rois,
                       size_t poolHeight, size_t poolWidth,
                       float spatial_scale) ;


  /* Specializations: CPU, float */
  // not yet implemented

  /* Specializations: GPU, float */
#if ENABLE_GPU
  template<> vl::Error
  roipooling_max_forward<vl::GPU, float>(float* pooled,
                                      float* argmax,
                                      float const* data,
                                      float const* rois,
                                      size_t height, size_t width, size_t channels, 
                                      size_t num_rois,
                                      size_t poolHeight, size_t poolWidth,
                                      float spatial_scale) ;

  template<> vl::Error
  roipooling_max_backward<vl::GPU, float>(float* derData,
                                       float const* data,
                                       float const* rois,
                                       float const* derPooled,
                                       float const* argmax,
                                       size_t height, size_t width, size_t channels, 
                                       size_t num_data, size_t num_rois,
                                       size_t poolHeight, size_t poolWidth,
                                       float spatial_scale) ;

#endif

} }

#endif /* defined(VL_NNROIPOOLING_H) */
