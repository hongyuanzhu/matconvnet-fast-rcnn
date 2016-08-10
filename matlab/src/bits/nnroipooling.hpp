// @file nnroipooling.hpp
// @brief ROI Pooling 
// @author Tuan-Hung Vu

/*
Copyright (C) 2015-16 Tuan-Hung Vu.
All rights reserved.

This file is made available under the terms of the BSD license.
*/

#ifndef __vl__nnroipooling__
#define __vl__nnroipooling__

#include "data.hpp"
#include <stdio.h>

namespace vl {

  enum PoolingMethod { vlPoolingMax, vlPoolingAverage } ;

  vl::Error
  nnroipooling_forward(vl::Context& context,
                    vl::Tensor output,
                    vl::Tensor argmax,
                    vl::Tensor data,
                    vl::Tensor rois,
                    PoolingMethod method,
                    size_t poolHeight, size_t poolWidth,
                    float spatial_scale) ;

  vl::Error
  nnroipooling_backward(vl::Context& context,
                     vl::Tensor derData,
                     vl::Tensor data,
                     vl::Tensor rois,
                     vl::Tensor derOutput,
                     vl::Tensor argmax,
                     PoolingMethod method,
                     size_t poolHeight, size_t poolWidth,
                     float spatial_scale) ;
}

#endif /* defined(__vl__nnroipooling__) */
