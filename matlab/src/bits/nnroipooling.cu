// @file nnroipooling.cu
// @brief ROI pooling
// @author Tuan-Hung Vu

/*
Copyright (C) 2015-16 Tuan-Hung Vu.
All rights reserved.

This file is made available under the terms of the BSD license.
*/

#include "nnroipooling.hpp"
#include "impl/roipooling.hpp"

#if ENABLE_GPU
#include "datacu.hpp"
#endif

#include <assert.h>

using namespace vl ;

/* ---------------------------------------------------------------- */
/*                                             nnroipooling_forward */
/* ---------------------------------------------------------------- */

Error
vl::nnroipooling_forward(vl::Context& context,
                      vl::Tensor output,
                      vl::Tensor argmax,
                      vl::Tensor data,
                      vl::Tensor rois,
                      PoolingMethod method,
                      size_t poolHeight, size_t poolWidth,
                      float spatial_scale)
{
  Error status = vlSuccess ;
  switch (output.getDeviceType()) {
    default:
      assert(false) ;
      return vl::vlErrorUnknown ;

    case vl::CPU:
      //mexErrMsgTxt("ROI pooling layer on CPU has not yet implemented.") ;
      break ;

#ifdef ENABLE_GPU
    case vl::GPU:
      switch (method) {
        default:
          assert(false) ;
          return vl::vlErrorUnknown ;
        case vl::vlPoolingMax:
          status = vl::impl::roipooling_max_forward<GPU,float>
          ((float*)output.getMemory(),
           (float*)argmax.getMemory(),
           (float const*)data.getMemory(),
           (float const*)rois.getMemory(),
           data.getHeight(), data.getWidth(), data.getDepth(), rois.getWidth(),
           poolHeight, poolWidth,
           spatial_scale) ;
          break;
      }
      if (status == vlErrorCuda) {
        context.setError(context.getCudaHelper().catchCudaError("roipooling_*_forward")) ;
      }
      break ;
#endif
  }
  return context.passError(status, "nnroipooling_forward: ") ;
}

/* ---------------------------------------------------------------- */
/*                                            nnroipooling_backward */
/* ---------------------------------------------------------------- */

Error
vl::nnroipooling_backward(vl::Context& context,
                       vl::Tensor derData,
                       vl::Tensor data,
                       vl::Tensor rois,
                       vl::Tensor derOutput,
                       vl::Tensor argmax,
                       PoolingMethod method,
                       size_t poolHeight, size_t poolWidth,
                       float spatial_scale)
{
  vl::Error status = vlSuccess ;
  switch (derData.getDeviceType()) {
    default:
      assert(false) ;
      return vl::vlErrorUnknown ;

    case vl::CPU:
      //mexErrMsgTxt("ROI pooling layer on CPU has not yet implemented.") ;
      break ;

#if ENABLE_GPU
    case vl::GPU:
      switch (method) {
        default:
          assert(false) ;
          return vl::vlErrorUnknown ;
        case vl::vlPoolingMax:
          status = vl::impl::roipooling_max_backward<GPU,float>
          ((float*)derData.getMemory(),
           (float const*)data.getMemory(),
           (float const*)rois.getMemory(),
           (float const*)derOutput.getMemory(),
           (float const*)argmax.getMemory(),
           derData.getHeight(), derData.getWidth(), derData.getDepth(), derData.getSize(),
           rois.getWidth(),
           poolHeight, poolWidth,
           spatial_scale) ;
          break ;
      }
      if (status == vlErrorCuda) {
        context.setError(context.getCudaHelper().catchCudaError("roipooling_*_backward: ")) ;
      }
      break ;
#endif
  }
  return context.passError(status, "nnroipooling_backward: ") ;
}
