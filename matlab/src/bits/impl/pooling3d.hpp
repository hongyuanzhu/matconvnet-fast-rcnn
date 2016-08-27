// @file pooling3d.hpp
// @brief 3D Pooling block implementation
// @author Tuan-Hung VU

/*
Copyright (C) 2016 Tuan-Hung VU
All rights reserved.
*/

#ifndef VL_POOLING3D_H
#define VL_POOLING3D_H

#include "../data.hpp"
#include <cstddef>

namespace vl { namespace impl {

  template<vl::Device dev, typename type>
  struct pooling3d_max {
    typedef type data_type ;

    static vl::Error
    forward(data_type* output,
            data_type const* data,
            size_t height, size_t width, size_t length, size_t depth,
            size_t poolHeight, size_t poolWidth, size_t poolLength,
            size_t strideY, size_t strideX, size_t strideL,
            size_t padTop, size_t padBottom, size_t padLeft, size_t padRight, size_t padLength) ;

    static vl::Error
    backward(data_type* derData,
             data_type const* data,
             data_type const* derOutput,
             size_t height, size_t width, size_t length, size_t depth,
             size_t poolHeight, size_t poolWidth, size_t poolLength,
             size_t strideY, size_t strideX, size_t strideL,
             size_t padTop, size_t padBottom, size_t padLeft, size_t padRight, size_t padLength) ;
  } ;

  template<vl::Device dev, typename type>
  struct pooling3d_average {
    typedef type data_type ;

    static vl::Error
    forward(data_type* output,
            data_type const* data,
            size_t height, size_t width, size_t length, size_t depth,
            size_t poolHeight, size_t poolWidth, size_t poolLength,
            size_t strideY, size_t strideX, size_t strideL,
            size_t padTop, size_t padBottom, size_t padLeft, size_t padRight, size_t padLength) ;

    static vl::Error
    backward(type* derData,
             type const* derOutput,
             size_t height, size_t width, size_t length, size_t depth,
             size_t poolHeight, size_t poolWidth, size_t poolLength,
             size_t strideY, size_t strideX, size_t strideL,
             size_t padTop, size_t padBottom, size_t padLeft, size_t padRight, size_t padLength) ;
  } ;

} }

#endif /* defined(VL_POOLING3D_H) */
