// @file nnpooling3d.hpp
// @brief 3D Pooling block
// @author Tuan-Hung VU

/*
Copyright (C) 2016 Tuan-Hung VU
All rights reserved.
*/

#ifndef __vl__nnpooling3d__
#define __vl__nnpooling3d__

#include "data.hpp"
#include <stdio.h>

namespace vl {

  enum PoolingMethod { vlPoolingMax, vlPoolingAverage } ;

  vl::Error
  nnpooling3d_forward(vl::Context& context,
                    vl::Tensor output,
                    vl::Tensor data,
                    PoolingMethod method,
                    int poolHeight, int poolWidth, int poolLength,
                    int strideY, int strideX, int strideL,
                    int padTop, int padBottom,
                    int padLeft, int padRight, int padLength) ;

  vl::Error
  nnpooling3d_backward(vl::Context& context,
                     vl::Tensor derData,
                     vl::Tensor data,
                     vl::Tensor derOutput,
                     PoolingMethod method,
                     int poolHeight, int poolWidth, int poolLength,
                     int strideY, int strideX, int strideL,
                     int padTop, int padBottom,
                     int padLeft, int padRight, int padLength) ;
}

#endif /* defined(__vl__nnpooling3d__) */
