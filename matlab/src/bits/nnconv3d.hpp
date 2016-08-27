// @file nnconv3d.cu
// @brief 3D Convolution block
// @author Tuan-Hung VU

/*
Copyright (C) 2016 Tuan-Hung VU

*/

#ifndef __vl__nnconv3d__
#define __vl__nnconv3d__

#include "data.hpp"

namespace vl {

  vl::Error
  nnconv3d_forward(vl::Context& context,
                 vl::Tensor output, double outputMult,
                 vl::Tensor data, double dataMult,
                 vl::Tensor filters,
                 vl::Tensor biases,
                 int strideY, int strideX, int strideT,
                 int padTop, int padBottom,
                 int padLeft, int padRight, int padT) ;

  vl::Error
  nnconv3d_backward(vl::Context& context,
                  vl::Tensor derData,
                  vl::Tensor derFilters,
                  vl::Tensor derBiases,
                  vl::Tensor data,
                  vl::Tensor filters,
                  vl::Tensor derOutput,
                  int strideY, int strideX, int strideT,
                  int padTop, int padBottom,
                  int padLeft, int padRight, int padT) ;
}


#endif /* defined(__vl__nnconv3d__) */
