// @file vol2row_gpu.cu
// @brief Stack voxels as matrix rows (GPU)
// @author Tuan-Hung Vu

/*
Copyright (C) 2016 Tuan-Hung Vu
All rights reserved.
*/

#include "vol2row.hpp"
#include "../datacu.hpp"
#include <iostream>

using namespace vl ;

/* ---------------------------------------------------------------- */
/*                                                           vol2row */
/* ---------------------------------------------------------------- */

template <typename T> __global__ void
vol2row_gpu_kernel(T* stacked,
                  T const* data,
                  const int numPatchesX,
                  const int numPatchesY,
                  const int numPatchesZ,
                  const int numPatchSlices,
                  const int width,
                  const int height,
                  const int length,
                  const int windowWidth,
                  const int windowHeight,
                  const int windowLength,
                  const int strideX,
                  const int strideY,
                  const int strideT,
                  const int padLeft,
                  const int padTop,
                  const int padT)
{
  /* each kernel copies the pixels in an voxel for one channel */
  int index = threadIdx.x + blockIdx.x * blockDim.x ;
  if (index < numPatchSlices) {
    /*
     get the patch slice (x,y,z) to copy
     */

    int x = index ;
    int y = x / numPatchesX ;
    int z = y / numPatchesY ;
    int c = z / numPatchesZ ;
    x %= numPatchesX ;
    y %= numPatchesY ;
    z %= numPatchesZ ;
            
    /*
     pick the top-left corer of the voxel in the input
     */
    int x_data = x * strideX - padLeft ;
    int y_data = y * strideY - padTop ;
    int z_data = z * strideT - padT ;
    data += ((c * length + z_data) * height + y_data) * width + x_data ;

    /*
     pick the column of the stacked image which contains this patch,
     and move down along the column at the beginning of the patch slice
     */
    int patchSliceOffset = (windowWidth*windowHeight*windowLength) * c ;
    stacked += (numPatchesY * (numPatchesZ * patchSliceOffset + z) + y) * numPatchesX + x ;

    /*
     copy the voxel slice
     */
    for (int t = 0 ; t < windowLength ; ++t) {
        for (int v = 0 ; v < windowHeight ; ++v) {
          for (int u = 0 ; u < windowWidth ; ++u) {
          if (y_data + v >= 0 &&
              y_data + v < height &&
              x_data + u >= 0 &&
              x_data + u < width &&
              z_data + t >= 0 &&
              z_data + t < length) {
            *stacked = data[(t * height + v) * width + u] ;
          } else {
            *stacked = 0 ;
          }
          stacked += (numPatchesX*numPatchesY*numPatchesZ) ;
        }
      }
    }
  }
}

template <typename T> static inline cudaError_t
vol2row_gpu(T* stacked,
           T const* data,
           size_t width,
           size_t height,
           size_t length,
           size_t depth,
           size_t windowWidth,
           size_t windowHeight,
           size_t windowLength,
           size_t strideX,
           size_t strideY,
           size_t strideT,
           size_t padLeft,
           size_t padRight,
           size_t padTop,
           size_t padBottom,
           size_t padT)
{
  /* Each kernel instance copies a feature dimension of a voxel */

  int numPatchesX = (width + (padLeft + padRight) - windowWidth)/strideX + 1 ;
  int numPatchesY = (height + (padTop + padBottom) - windowHeight)/strideY + 1 ;
  int numPatchesZ = (length + (padT + padT) - windowLength)/strideT + 1 ;
  int numPatchSlices = numPatchesX * numPatchesY * numPatchesZ * depth ;

  vol2row_gpu_kernel<T>
  <<< divideUpwards(numPatchSlices, VL_CUDA_NUM_THREADS), VL_CUDA_NUM_THREADS >>>
  (stacked,
   data,
   numPatchesX,
   numPatchesY,
   numPatchesZ,
   numPatchSlices,
   width, height, length,
   windowWidth, windowHeight, windowLength,
   strideX, strideY, strideT,
   padLeft, padTop, padT) ;

  return cudaPeekAtLastError() ;
}


template <> vl::Error
vl::impl::vol2row<vl::GPU, float>(vl::Context& context,
                                 float* stacked,
                                 float const* data,
                                 size_t height, size_t width, size_t length, size_t depth,
                                 size_t windowHeight, size_t windowWidth, size_t windowLength,
                                 size_t strideY, size_t strideX, size_t strideT,
                                 size_t padTop, size_t padBottom, size_t padLeft, size_t padRight, size_t padT)
{
  int status ;
  status = vol2row_gpu<float>(stacked, data,
                             height, width, length, depth,
                             windowHeight, windowWidth, windowLength,
                             strideY, strideX, strideT,
                             padTop, padBottom, padLeft, padRight, padT) ;
  return (status == cudaSuccess) ? vl::vlSuccess : vl::vlErrorCuda ;
}


/* ---------------------------------------------------------------- */
/*                                                           row2vol */
/* ---------------------------------------------------------------- */

template <typename T>
__global__ void row2vol_gpu_kernel(T* data,
                                  T const* stacked,
                                  const int numPatchesX,
                                  const int numPatchesY,
                                  const int numPatchesZ,
                                  const int dataVolume,
                                  const int width,
                                  const int height,
                                  const int length,
                                  const int depth,
                                  const int windowWidth,
                                  const int windowHeight,
                                  const int windowLength,
                                  const int strideX,
                                  const int strideY,
                                  const int strideT,
                                  const int padLeft,
                                  const int padTop,
                                  const int padT)
{
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index < dataVolume)
  {
    T accumulator = 0 ;

    int x_data = index ;
    int y_data = x_data / width ;
    int z_data = y_data / height ;
    int c = z_data / length ;
    x_data %= width ;
    y_data %= height ;
    z_data %= length ;

    int dx = x_data + padLeft - windowWidth ;
    int dy = y_data + padTop - windowHeight ;
    int dz = z_data + padT - windowLength ;
    
    int x1 = (dx >= 0) ? dx/strideX + 1 : 0 ;
    int y1 = (dy >= 0) ? dy/strideY + 1 : 0 ;
    int z1 = (dz >= 0) ? dz/strideT + 1 : 0 ;
    
    int x2 = min((x_data + padLeft) / strideX, numPatchesX - 1) ;
    int y2 = min((y_data + padTop) / strideY, numPatchesY - 1) ;
    int z2 = min((z_data + padT) / strideT, numPatchesZ - 1) ;

    /* voxel indexing
    u(x) = x_data - (x * strideX - padLeft)
    v(y) = y_data - (y * strideY - padTop)
    t(z) = z_data - (z * strideT - padT)
    
    stackedIndex(x,y) =
     ((z * numPatchesY + y) * numPatchesX + x) +  
     (((c * windowLength + t(z)) * windowHeight + v(y)) * windowWidth + u(x)) * 
     (numPatchesX*numPatchesY*numPatchesZ)
     
    = ((z * numPatchesY + y) * numPatchesX + x) + 
     (((c * windowLength + z_data - (z * strideT - padT)) * windowHeight + y_data - (y * strideY - padTop)) * windowWidth + x_data - (x * strideX - padLeft))
     * (numPatchesX*numPatchesY*numPatchesZ)
     
    = x * (1 - strideX*numPatchesX*numPatchesY*numPatchesZ)
    + y * (1 - strideY*numPatchesY*numPatchesZ*windowWidth) * numPatchesX
    + z * (1 - strideT*numPatchesZ*windowWidth*windowHeight) * numPatchesY * numPatchesX
    + (( (c * windowLength + z_data + padT) * windowHeight + y_data + padTop) * windowWidth + (x_data + padLeft)) * (numPatchesX*numPatchesY*numPatchesZ)   
     */

    int deltax = (1 - strideX * numPatchesX * numPatchesY * numPatchesZ);
    int deltay = (1 - strideY * numPatchesY * numPatchesZ * windowWidth) * numPatchesX ;
    int deltaz = (1 - strideT * numPatchesZ * windowWidth * windowHeight) * numPatchesY * numPatchesX;
    
    stacked += (( (c * windowLength + z_data + padT) * windowHeight + y_data + padTop) * windowWidth + (x_data + padLeft)) * (numPatchesX*numPatchesY*numPatchesZ) ;

    for (int z = z1 ; z <= z2 ; ++ z) {
      for (int y = y1 ; y <= y2 ; ++ y) {
        for (int x = x1 ; x <= x2 ; ++ x) {
          accumulator += stacked[z*deltaz + y * deltay + x * deltax];
        }
      }
    }
    data[index] = accumulator;
  }
}

template <typename T> static inline cudaError_t
row2vol_gpu(T* data,
           T const* stacked,
           size_t width,
           size_t height,
           size_t length,
           size_t depth,
           size_t windowWidth,
           size_t windowHeight,
           size_t windowLength,
           size_t strideX,
           size_t strideY,
           size_t strideT,
           size_t padLeft,
           size_t padRight,
           size_t padTop,
           size_t padBottom,
           size_t padT)
{
  /*
   Each kernel integrates all contributions to a particular element
   of data.
   */

  int numPatchesX = (width + (padLeft + padRight) - windowWidth)/strideX + 1 ;
  int numPatchesY = (height + (padTop + padBottom) - windowHeight)/strideY + 1 ;
  int numPatchesZ = (length + (padT + padT) - windowLength)/strideT + 1 ;
  int dataVolume = width * height * length * depth ;

  row2vol_gpu_kernel<T>
  <<< divideUpwards(dataVolume, VL_CUDA_NUM_THREADS), VL_CUDA_NUM_THREADS >>>
  (data,
   stacked,
   numPatchesX,
   numPatchesY,
   numPatchesZ,
   dataVolume,
   width, height, length, depth,
   windowWidth, windowHeight, windowLength,
   strideX, strideY, strideT,
   padLeft, padTop, padT) ;

  return cudaPeekAtLastError() ;
}

template <> vl::Error
vl::impl::row2vol<vl::GPU, float>(vl::Context& context,
                                 float* data,
                                 float const* stacked,
                                 size_t height, size_t width, size_t length, size_t depth,
                                 size_t windowHeight, size_t windowWidth, size_t windowLength,
                                 size_t strideY, size_t strideX, size_t strideT,
                                 size_t padTop, size_t padBottom, size_t padLeft, size_t padRight, size_t padT)
{
  int status ;
  status = row2vol_gpu<float>(data, stacked,
                             height, width, length, depth,
                             windowHeight, windowWidth, windowLength,
                             strideY, strideX, strideT,
                             padTop, padBottom, padLeft, padRight, padT) ;
  return (status == cudaSuccess) ? vl::vlSuccess : vl::vlErrorCuda ;
}
