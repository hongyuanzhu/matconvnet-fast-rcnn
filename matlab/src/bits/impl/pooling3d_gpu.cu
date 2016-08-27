// @file pooling3d_gpu.cu
// @brief 3D Pooling block implementation (GPU)
// @author Tuan-Hung VU

/*
Copyright (C) 2016 Tuan-Hung VU
All rights reserved.
*/

#include "pooling3d.hpp"
#include "../datacu.hpp"
#include <assert.h>
#include <float.h>
#include <sm_20_atomic_functions.h>
#include "pooling_gpu.cu"

/* ---------------------------------------------------------------- */
/*                                              pooling3d_max_forward */
/* ---------------------------------------------------------------- */

template<typename T> __global__ void
pooling3d_max_kernel
(T* pooled,
 const T* data,
 const int pooledWidth,
 const int pooledHeight,
 const int pooledLength,
 const int pooledVolume,
 const int width,
 const int height,
 const int length,
 const int poolWidth,
 const int poolHeight,
 const int poolLength,
 const int strideX,
 const int strideY,
 const int strideL,
 const int padLeft,
 const int padTop,
 const int padLength)
{
  int pooledIndex = threadIdx.x + blockIdx.x * blockDim.x;
  if (pooledIndex < pooledVolume) {
    int px = pooledIndex ;
    int py = px / pooledWidth ;
    int pz = py / pooledHeight ;
    int pc = pz / pooledLength ;

    px %= pooledWidth ;
    py %= pooledHeight ;
    pz %= pooledLength ;

    data += pc * (width*height*length) ;

    int x1 = px * strideX - padLeft ;
    int y1 = py * strideY - padTop ;
    int z1 = pz * strideL - padLength ;

    int x2 = min(x1 + poolWidth, width) ;
    int y2 = min(y1 + poolHeight, height) ;
    int z2 = min(z1 + poolLength, length) ;

    x1 = max(x1, 0) ;
    y1 = max(y1, 0) ;
    z1 = max(z1, 0) ;

    T bestValue = data[(z1 * height + y1) * width + x1] ;
    for (int z = z1 ; z < z2 ; ++z) {
      for (int y = y1 ; y < y2 ; ++y) {
        for (int x = x1 ; x < x2 ; ++x) {
          bestValue = max(bestValue, data[(z * height + y) * width + x]) ;
        }
      }
    }
    pooled[pooledIndex] = bestValue ;
  }
}

/* ---------------------------------------------------------------- */
/*                                          pooling3d_average_forward */
/* ---------------------------------------------------------------- */

template<typename T> __global__ void
pooling3d_average_kernel
(T* pooled,
 const T* data,
 const int pooledWidth,
 const int pooledHeight,
 const int pooledLength,
 const int pooledVolume,
 const int width,
 const int height,
 const int length,
 const int poolWidth,
 const int poolHeight,
 const int poolLength,
 const int strideX,
 const int strideY,
 const int strideL,
 const int padLeft,
 const int padTop,
 const int padLength)
{
  /* pooledIndex = x + y * pooledWidth + z * (pooledWidth * pooledHeight) */
  int pooledIndex = threadIdx.x + blockIdx.x * blockDim.x;
  if (pooledIndex < pooledVolume) {
    int px = pooledIndex ;
    int py = px / pooledWidth ;
    int pz = py / pooledHeight ;
    int pc = pz / pooledLength ;

    px %= pooledWidth ;
    py %= pooledHeight ;
    pz %= pooledLength ;

    int x1 = px * strideX - padLeft ;
    int y1 = py * strideY - padTop ;
    int z1 = pz * strideL - padLength ;

    int x2 = min(x1 + poolWidth, width) ;
    int y2 = min(y1 + poolHeight, height) ;
    int z2 = min(z1 + poolLength, length) ;

    x1 = max(x1, 0) ;
    y1 = max(y1, 0) ;
    z1 = max(z1, 0) ;

    data += pc * (width*height*length) ;

    T accum = 0;
    T poolSize = (z2 - z1)*(y2 - y1)*(x2 - x1);
    for (int z = z1 ; z < z2 ; ++z) {
      for (int y = y1 ; y < y2 ; ++y) {
        for (int x = x1 ; x < x2 ; ++x) {
          accum += data[(z * height + y) * width + x] ;
        }
      }
    }
    pooled[pooledIndex] = accum / poolSize ;
  }
}

/* ---------------------------------------------------------------- */
/*                                             pooling3d_max_backward */
/* ---------------------------------------------------------------- */

#ifdef VLNN_CAFFELIKE_BPPOOL
// In order to be able to use this, BP would need to have access to both
// bottom data and pooled data (currently only passed bottom data...)
template <typename T> __global__ void
pooling3d_max_backward_with_pooled_data
(T* derData,
 const T* data,
 const T* pooled,
 const T* derPooled,
 const int nthreads,
 const int pooledWidth,
 const int pooledHeight,
 const int pooledLength,
 const int width,
 const int height,
 const int length,
 const int depth,
 const int poolWidth,
 const int poolHeight,
 const int poolLength,
 const int strideX,
 const int strideY,
 const int strideL)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < nthreads) {
    // find out the local index
    // find out the local offset
    int x = index % width;
    int y = (index / width) % height;
    int z = (index / width / height) % length;
    int c = (index / width / heigth / length) % depth;

    int pz1 = (z < poolLength) ? 0 : (z - poolLength) / strideL + 1;
    int pz2 = min(z / strideL + 1, pooledLength);

    int py1 = (y < poolHeight) ? 0 : (y - poolHeight) / strideY + 1;
    int py2 = min(y / strideY + 1, pooledHeight);

    int px1 = (x < poolWidth) ? 0 : (x - poolWidth) / strideX + 1;
    int px2 = min(x / strideX + 1, pooledWidth); 

    T gradient = 0;
    T datum = data[(( c * length + z) * height + y) * width + x];
    pooled += c * pooledHeight * pooledWidth * pooledLength;
    dzdy += c * pooledHeight * pooledWidth * pooledLength;
    for (int pz = pz1; pz < pz2; ++pz) {
      for (int py = py1; py < py2; ++py) {
        for (int px = px1; px < px2; ++px) {
          gradient += dzdy[(pz * poolHeight + py) * pooledWidth + px] *
          (datum == pooled[(pz * poolHeight + py) * pooledWidth + px]);
        }
      }
    }
    dzdx[index] = gradient;
  }
}
#endif

//#ifndef __vl__atomicAdd_double__
//#define __vl__atomicAdd_double__ 
//// an implementation of atomicAdd() for double (really slow)
//__device__ double atomicAdd(double* address, double val)
//{
//  unsigned long long int* address_as_ull = (unsigned long long int*)address;
//  unsigned long long int old = *address_as_ull, assumed;
//  do {
//    assumed = old;
//    old = atomicCAS(address_as_ull, assumed,
//                    __double_as_longlong(val +
//                                         __longlong_as_double(assumed)));
//  } while (assumed != old);
//  return __longlong_as_double(old);
//}
//#endif

template<typename T> __global__ void
pooling3d_max_backward_kernel
(T* derData,
 const T* data,
 const T* derPooled,
 const int pooledWidth,
 const int pooledHeight,
 const int pooledLength,
 const int pooledVolume,
 const int width,
 const int height,
 const int length,
 const int poolWidth,
 const int poolHeight,
 const int poolLength,
 const int strideX,
 const int strideY,
 const int strideL,
 const int padLeft,
 const int padTop,
 const int padLength)
{
  int pooledIndex = threadIdx.x + blockIdx.x * blockDim.x;
  if (pooledIndex < pooledVolume) {
    int px = pooledIndex ;
    int py = px / pooledWidth ;
    int pz = py / pooledHeight ;
    int pc = pz / pooledLength ;

    px %= pooledWidth ;
    py %= pooledHeight ;
    pz %= pooledLength ;

    data += pc * (width*height*length) ;
    derData += pc * (width*height*length) ;

    int x1 = px * strideX - padLeft ;
    int y1 = py * strideY - padTop ;
    int z1 = pz * strideL - padLength ;

    int x2 = min(x1 + poolWidth, width) ;
    int y2 = min(y1 + poolHeight, height) ;
    int z2 = min(z1 + poolLength, length);

    x1 = max(x1, 0) ;
    y1 = max(y1, 0) ;
    z1 = max(z1, 0) ;

    int bestIndex = (z1 * height + y1) * width + x1 ;
    T bestValue = data[bestIndex] ;
    for (int z = z1 ; z < z2 ; ++z) {
      for (int y = y1 ; y < y2 ; ++y) {
        for (int x = x1 ; x < x2 ; ++x) {
          int index = (z * height + y) * width + x ;
          T value = data[index] ;
          if (value > bestValue) {
            bestValue = value ;
            bestIndex = index ;
          }
        }
      }
    }
    /*
     This is bad, but required to eliminate a race condition when writing
     to bottom_diff.
     Caffe goes the other way around, but requrires remembering the layer
     output, or the maximal indexes.
     atomicAdd(add, val)
     */
    atomicAdd(derData + bestIndex, derPooled[pooledIndex]) ;
  }
}

/* ---------------------------------------------------------------- */
/*                                         pooling3d_average_backward */
/* ---------------------------------------------------------------- */

template <typename T> __global__ void
pooling3d_average_backward_kernel(T* derData,
                                const T* derPooled,
                                const int nthreads,
                                const int pooledWidth,
                                const int pooledHeight,
                                const int pooledLength,
                                const int width,
                                const int height,
                                const int length,
                                const int depth,
                                const int poolWidth,
                                const int poolHeight,
                                const int poolLength,
                                const int strideX,
                                const int strideY,
                                const int strideL,
                                const int padLeft,
                                const int padTop,
                                const int padLength)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < nthreads) {
    /* To understand the logic of this piece of code see the
     comments to of the row2im backward kernel */
    int x_data = index ;
    int y_data = x_data / width ;
    int z_data = y_data / height ;
    int c = z_data / length;

    x_data %= width ;
    y_data %= height ;
    z_data %= length ;

    int dx = x_data + padLeft - poolWidth ;
    int dy = y_data + padTop - poolHeight ;
    int dz = z_data + padLength - poolLength ;

    int px1 = (dx >= 0) ? dx/strideX + 1 : 0 ;
    int py1 = (dy >= 0) ? dy/strideY + 1 : 0 ;
    int pz1 = (dz >= 0) ? dz/strideL + 1 : 0 ;

    int px2 = min((x_data + padLeft) / strideX, pooledWidth - 1) ;
    int py2 = min((y_data + padTop) / strideY, pooledHeight - 1) ;
    int pz2 = min((z_data + padLength) / strideL, pooledLength - 1);

    T accumulator = 0 ;
    derPooled += c * pooledHeight * pooledWidth * pooledLength;
   
    for (int pz = pz1 ;  pz <= pz2; ++pz) {
      for (int py = py1 ; py <= py2 ; ++py) {
        for (int px = px1 ; px <= px2 ; ++px) {
          int x1 = px * strideX - padLeft ;
          int y1 = py * strideY - padTop ;
          int z1 = pz * strideL - padLength ;

          int x2 = min(x1 + poolWidth, width) ;
          int y2 = min(y1 + poolHeight, height) ;
          int z2 = min(z1 + poolLength, length) ;

          x1 = max(x1, 0) ;
          y1 = max(y1, 0) ;
          z1 = max(z1, 0) ;

          T poolSize = (z2 - z1)*(y2 - y1) * (x2 - x1);
          accumulator += derPooled[(pz * pooledHeight + py) * pooledWidth + px] / poolSize ;
        }
      }
    }
    derData[index] = accumulator ;
  }
}

/* ---------------------------------------------------------------- */
/*                                                        Interface */
/* ---------------------------------------------------------------- */

namespace vl { namespace impl {

  template <typename type>
  struct pooling3d_max<vl::GPU, type>
  {
    static vl::Error
    forward(type* pooled,
            type const* data,
            size_t height, size_t width, size_t length, size_t depth,
            size_t poolHeight, size_t poolWidth, size_t poolLength,
            size_t strideY, size_t strideX, size_t strideL,
            size_t padTop, size_t padBottom,
            size_t padLeft, size_t padRight, size_t padLength)
    {
      int pooledWidth = (width + (padLeft+padRight) - poolWidth)/strideX + 1 ;
      int pooledHeight = (height + (padTop+padBottom) - poolHeight)/strideY + 1 ;
      int pooledLength = (length + (padLength+padLength) - poolLength)/strideL + 1;
      int pooledVolume = pooledWidth * pooledHeight * pooledLength * depth ;

      pooling3d_max_kernel<type>
      <<< divideUpwards(pooledVolume, VL_CUDA_NUM_THREADS), VL_CUDA_NUM_THREADS >>>
      (pooled, data,
       pooledHeight, pooledWidth, pooledLength, pooledVolume,
       height, width, length,
       poolHeight, poolWidth, poolLength,
       strideY, strideX, strideL,
       padTop, padLeft, padLength);

      cudaError_t status = cudaPeekAtLastError() ;
      return (status == cudaSuccess) ? vl::vlSuccess : vl::vlErrorCuda ;
    }

    static vl::Error
    backward(type* derData,
             type const* data,
             type const* derOutput,
             size_t height, size_t width, size_t length, size_t depth,
             size_t poolHeight, size_t poolWidth, size_t poolLength,
             size_t strideY, size_t strideX, size_t strideL,
             size_t padTop, size_t padBottom,
             size_t padLeft, size_t padRight, size_t padLength)
    {
      int pooledWidth = (width + (padLeft+padRight) - poolWidth)/strideX + 1 ;
      int pooledHeight = (height + (padTop+padBottom) - poolHeight)/strideY + 1 ;
      int pooledLength = (length + (padLength+padLength) - poolLength)/strideL + 1;
      int pooledVolume = pooledWidth * pooledHeight * pooledLength * depth ;

      pooling3d_max_backward_kernel<type>
      <<< divideUpwards(pooledVolume, VL_CUDA_NUM_THREADS), VL_CUDA_NUM_THREADS >>>
      (derData, data, derOutput,
       pooledHeight, pooledWidth, pooledLength, pooledVolume,
       height, width, length,
       poolHeight, poolWidth, poolLength,
       strideY, strideX, strideL,
       padTop, padLeft, padLength);

      cudaError_t status = cudaPeekAtLastError() ;
      return (status == cudaSuccess) ? vl::vlSuccess : vl::vlErrorCuda ;
    }
  } ; // pooling_max

  template <typename type>
  struct pooling3d_average<vl::GPU, type>
  {

    static vl::Error
    forward(type* pooled,
            type const* data,
            size_t height, size_t width, size_t length, size_t depth,
            size_t poolHeight, size_t poolWidth, size_t poolLength,
            size_t strideY, size_t strideX, size_t strideL,
            size_t padTop, size_t padBottom, size_t padLeft, size_t padRight, size_t padLength)
    {
      int pooledWidth = (width + (padLeft+padRight) - poolWidth)/strideX + 1 ;
      int pooledHeight = (height + (padTop+padBottom) - poolHeight)/strideY + 1 ;
      int pooledLength = (length + (padLength+padLength) - poolLength)/strideL + 1;
      int pooledVolume = pooledWidth * pooledHeight * pooledLength * depth ;

      pooling3d_average_kernel<type>
      <<< divideUpwards(pooledVolume, VL_CUDA_NUM_THREADS), VL_CUDA_NUM_THREADS >>>
      (pooled, data,
       pooledHeight, pooledWidth, pooledLength, pooledVolume,
       height, width, length,
       poolHeight, poolWidth, poolLength,
       strideY, strideX, strideL,
       padTop, padLeft, padLength);

      cudaError_t status = cudaPeekAtLastError() ;
      return (status == cudaSuccess) ? vl::vlSuccess : vl::vlErrorCuda ;
    }

    static vl::Error
    backward(type* derData,
             type const* derPooled,
             size_t height, size_t width, size_t length, size_t depth,
             size_t poolHeight, size_t poolWidth, size_t poolLength,
             size_t strideY, size_t strideX, size_t strideL,
             size_t padTop, size_t padBottom,
             size_t padLeft, size_t padRight, size_t padLength)
    {
      int pooledWidth = (width + (padLeft+padRight) - poolWidth)/strideX + 1 ;
      int pooledHeight = (height + (padTop+padBottom) - poolHeight)/strideY + 1 ;
      int pooledLength = (length + (padLength+padLength) - poolLength)/strideL + 1;
      int dataVolume = width * height * length * depth ;

      pooling3d_average_backward_kernel<type>
      <<< divideUpwards(dataVolume, VL_CUDA_NUM_THREADS), VL_CUDA_NUM_THREADS >>>
      (derData, derPooled,
       dataVolume,
       pooledHeight, pooledWidth, pooledLength,
       height, width, length, dataVolume,
       poolHeight, poolWidth, poolLength,
       strideY, strideX, strideL,
       padTop, padLeft, padLength);

      cudaError_t status = cudaPeekAtLastError() ;
      return (status == cudaSuccess) ? vl::vlSuccess : vl::vlErrorCuda ;
    }
  } ; // pooling_average

} } ; // namespace vl::impl

// Instantiations
template struct vl::impl::pooling3d_max<vl::GPU, float> ;
template struct vl::impl::pooling3d_average<vl::GPU, float> ;

#ifdef ENABLE_DOUBLE
template struct vl::impl::pooling3d_max<vl::GPU, double> ;
template struct vl::impl::pooling3d_average<vl::GPU, double> ;
#endif

