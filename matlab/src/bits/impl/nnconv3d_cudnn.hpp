// @file nnconv3d_cudnn.hpp
// @brief 3D Convolution block CuDNN-based implementation.
// @author Tuan-Hung VU

/*
Copyright (C) 2015-16 Tuan-Hung VU
All rights reserved.
*/

#ifndef __vl__nnconv3d_cudnn__
#define __vl__nnconv3d_cudnn__

#include "../data.hpp"
#include "cudnn.h"

namespace vl { namespace impl {

  template<vl::Type dataType>
  struct nnconv3d_cudnn
  {
    static vl::Error
    forward(Context& context,
            Tensor output, double outputMult,
            Tensor data, double dataMult,
            Tensor filters,
            Tensor biases,
            int strideX, int strideY, int strideT,
            int padLeft, int padRight,
            int padTop, int padBottom, int padT) ;

    static vl::Error
    backward(Context& context,
             Tensor derData,
             Tensor derFilters,
             Tensor derBiases,
             Tensor data,
             Tensor filters,
             Tensor derOutput,
             int strideX, int strideY, int strideT,
             int padLeft, int padRight,
             int padTop, int padBottom, int padT) ;
  } ;

} }
#endif /* defined(__vl__nnconv3d_cudnn__) */
