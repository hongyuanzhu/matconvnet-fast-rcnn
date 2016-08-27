// @file nnpooling3d.cu
// @brief 3D Pooling block
// @author Tuan-Hung VU

/*
Copyright (C) 2016 Tuan-Hung VU
All rights reserved.
*/

#include "nnpooling3d.hpp"
#include "impl/pooling3d.hpp"

#if ENABLE_GPU
#include "datacu.hpp"
#endif
#include <assert.h>

using namespace vl ;

/* ---------------------------------------------------------------- */
/*                                                nnpooling3d_forward */
/* ---------------------------------------------------------------- */

#define DISPATCH(deviceType, op, type) \
status = vl::impl::op<deviceType, type>::forward \
((type*)output.getMemory(), (type const*)data.getMemory(), \
data.getDimension(0), data.getDimension(1), data.getDimension(2), data.getDimension(3) * data.getDimension(4), \
poolHeight, poolWidth, poolLength, \
strideY, strideX, strideL, \
padTop, padBottom, \
padLeft, padRight, padLength) ;

#define DISPATCH2(deviceType, op) \
switch (dataType) { \
case vlTypeFloat : DISPATCH(deviceType, op, float) ; break ; \
IF_DOUBLE(case vlTypeDouble : DISPATCH(deviceType, op, double) ; break ;) \
default: assert(false) ; return vlErrorUnknown ; \
}

#define DISPATCH3(deviceType) \
switch (method) { \
case vlPoolingAverage : DISPATCH2(deviceType, pooling3d_average) ; break ; \
case vlPoolingMax : DISPATCH2(deviceType, pooling3d_max) ; break ; \
default: assert(false) ; return vlErrorUnknown ; \
}

vl::Error
vl::nnpooling3d_forward(vl::Context& context,
                      vl::Tensor output,
                      vl::Tensor data,
                      PoolingMethod method,
                      int poolHeight, int poolWidth, int poolLength,
                      int strideY, int strideX, int strideL,
                      int padTop, int padBottom,
                      int padLeft, int padRight, int padLength)
{
  vl::Error status = vlSuccess ;
  vl::Device deviceType = output.getDeviceType() ;
  vl::Type dataType = output.getDataType() ;

  switch (deviceType) {
    default:
      assert(false) ;
      return vl::vlErrorUnknown ;

    case vl::CPU:
      break ;

#ifdef ENABLE_GPU
    case vl::GPU:
      DISPATCH3(GPU) ;
      if (status == vlErrorCuda) {
        context.setError(context.getCudaHelper().catchCudaError(__func__)) ;
      }
      break ;
#endif
  }
  return context.passError(status, "nnpooling3d_forward") ;
}

/* ---------------------------------------------------------------- */
/*                                               nnpooling3d_backward */
/* ---------------------------------------------------------------- */

#undef DISPATCH
#undef DISPATCH2

// backward max and average want slightly differet argument lists

#define DISPATCH_pooling3d_average(deviceType, type) \
status = vl::impl::pooling3d_average<deviceType, type>::backward \
((type*)derData.getMemory(), (type const*)derOutput.getMemory(), \
derData.getDimension(0), derData.getDimension(1), derData.getDimension(2), derData.getDimension(3) * derData.getDimension(4), \
poolHeight, poolWidth, poolLength, \
strideY, strideX, strideL, \
padTop, padBottom, \
padLeft, padRight, padLength) ;

#define DISPATCH_pooling3d_max(deviceType, type) \
status = vl::impl::pooling3d_max<deviceType, type>::backward \
((type*)derData.getMemory(), (type const*)data.getMemory(), (type const*)derOutput.getMemory(), \
derData.getDimension(0), derData.getDimension(1), derData.getDimension(2), derData.getDimension(3) * derData.getDimension(4), \
poolHeight, poolWidth, poolLength, \
strideY, strideX, strideL, \
padTop, padBottom, \
padLeft, padRight, padLength) ;

#define DISPATCH2(deviceType, op) \
switch (dataType) { \
case vlTypeFloat : DISPATCH_ ## op (deviceType, float) ; break ; \
IF_DOUBLE(case vlTypeDouble : DISPATCH_ ## op (deviceType, double) ; break ;) \
default: assert(false) ; return vlErrorUnknown ; \
}

vl::Error
vl::nnpooling3d_backward(Context& context,
                       Tensor derData,
                       Tensor data,
                       Tensor derOutput,
                       PoolingMethod method,
                       int poolHeight, int poolWidth, int poolLength,
                       int strideY, int strideX, int strideL,
                       int padTop, int padBottom,
                       int padLeft, int padRight, int padLength)
{
  vl::Error status = vlSuccess ;
  vl::Device deviceType = derOutput.getDeviceType() ;
  vl::Type dataType = derOutput.getDataType() ;

  switch (deviceType) {
    default:
      assert(false) ;
      return vl::vlErrorUnknown ;

    case vl::CPU:
      break ;

#if ENABLE_GPU
    case vl::GPU:
      DISPATCH3(vl::GPU) ;
      if (status == vlErrorCuda) {
        context.setError(context.getCudaHelper().catchCudaError("pooling3d_*::backward")) ;
      }
      break ;
#endif
  }
  return context.passError(status, "nnpooling3d_backward") ;
}
