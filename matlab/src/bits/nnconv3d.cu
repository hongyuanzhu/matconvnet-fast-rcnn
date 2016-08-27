// @file nnconv3d.cu
// @brief 3d Convolution block
// @author Tuan-Hung Vu

/*
Copyright (C) 2016 Tuan-Hung VU
*/

#include "nnconv3d.hpp"
#include "nnbias.hpp"
#include "impl/nnconv3d_blas.hpp"
#if ENABLE_CUDNN
#include "impl/nnconv3d_cudnn.hpp"
#endif
#include <assert.h>

using namespace vl ;

/* ---------------------------------------------------------------- */
/*                                                   nnconv_forward */
/* ---------------------------------------------------------------- */

/*
 for output: must have data and optional filters or biases
 */

vl::Error
vl::nnconv3d_forward(Context& context,
                   Tensor output, double outputMult,
                   Tensor data, double dataMult,
                   Tensor filters,
                   Tensor biases,
                   int strideY, int strideX, int strideT,
                   int padTop, int padBottom,
                   int padLeft, int padRight, int padT)
{
  vl::Error status = vlSuccess ;
  switch (output.getDeviceType()) {
    default:
      assert(false) ;
      status = vl::vlErrorUnknown ;
      break ;

    case vl::CPU:
      break ;

#if ENABLE_GPU
    case vl::GPU:
#if ENABLE_CUDNN
      if (context.getCudaHelper().getCudnnEnabled()) {

        status = vl::impl::nnconv3d_cudnn<vlTypeFloat>::forward
        (context,
         output, outputMult,
         data, dataMult,
         filters, biases,
         strideY, strideX, strideT,
         padTop, padBottom,
         padLeft, padRight, padT) ;

        if (status == vl::vlSuccess) { return status ; }
        if (status != vl::vlErrorUnsupported) { goto done ; }
        /* this case was not supported by CUDNN -- fallback */
      }
#endif
      status = vl::impl::nnconv3d_forward_blas<GPU,vlTypeFloat>
      (context,
       output, outputMult,
       data, dataMult,
       filters, biases,
       strideY, strideX, strideT,
       padTop, padBottom,
       padLeft, padRight, padT) ;
      break ;
#endif
  }
#if ENABLE_CUDNN
done:
#endif
  return status ;
}

/* ---------------------------------------------------------------- */
/*                                                  nnconv_backward */
/* ---------------------------------------------------------------- */


/*
 for derBiases:  must have derOuptut
 for derData:    must have derData, derOutput and filters
 for derFilters: must have derFilters, derOutput and data
 */

vl::Error
vl::nnconv3d_backward(Context& context,
                    Tensor derData,
                    Tensor derFilters,
                    Tensor derBiases,
                    Tensor data,
                    Tensor filters,
                    Tensor derOutput,
                    int strideY, int strideX, int strideT,
                    int padTop, int padBottom,
                    int padLeft, int padRight, int padT)
{
  vl::Error status = vl::vlSuccess ;
  switch (derOutput.getDeviceType()) {
    default:
      assert(false) ;
      status = vl::vlErrorUnknown ;
      break ;

    case vl::CPU:
      break ;

#if ENABLE_GPU
    case vl::GPU:
#if ENABLE_CUDNN
      if (context.getCudaHelper().getCudnnEnabled()) {
        status = vl::impl::nnconv3d_cudnn<vlTypeFloat>::backward
        (context,
         derData, derFilters, derBiases,
         data, filters, derOutput,
         strideY, strideX, strideT,
         padTop, padBottom,
         padLeft, padRight, padT) ;
        if (status == vl::vlSuccess) { return status ; }
        if (status != vl::vlErrorUnsupported) { goto done ; }
        /* this case was not supported by CUDNN -- fallback */
      }
#endif
      status = vl::impl::nnconv3d_backward_blas<GPU,vlTypeFloat>
      (context,
       derData, derFilters, derBiases,
       data, filters, derOutput,
       strideY, strideX, strideT,
       padTop, padBottom,
       padLeft, padRight, padT) ;
      break;
#endif
  }
#if ENABLE_CUDNN
done:
#endif
  return status ;
}

