// @file vol2row.hpp
// @brief Stack voxels as matrix rows
// @author Tuan-Hung VU

/*
Copyright (C) 2016 Tuan-Hung VU
All rights reserved.
*/

#ifndef __vl__vol2row__
#define __vl__vol2row__

#include "../data.hpp"
#include <stddef.h>

namespace vl { namespace impl {

  template<vl::Device dev, typename type> vl::Error
  vol2row(vl::Context& context,
         type* stacked,
         type const* data,
         size_t height, size_t width, size_t length, size_t depth,
         size_t windowHeight, size_t windowWidth, size_t windowLength,
         size_t strideY, size_t strideX, size_t strideT,
         size_t padTop, size_t padBottom, size_t padLeft, size_t padRight, size_t padT) ;

  template<vl::Device dev, typename type> vl::Error
  row2vol(vl::Context& context,
         type* data,
         type const* stacked,
         size_t height, size_t width, size_t length, size_t depth,
         size_t windowHeight, size_t windowWidth, size_t windowLength,
         size_t strideY, size_t strideX, size_t strideT,
         size_t padTop, size_t padBottom, size_t padLeft, size_t padRight, size_t padT) ;
  
#if ENABLE_GPU
  template<> vl::Error
  vol2row<vl::GPU, float>(vl::Context& context,
                         float* stacked,
                         float const* data,
                         size_t height, size_t width, size_t length, size_t depth,
                         size_t windowHeight, size_t windowWidth, size_t windowLength,
                         size_t strideY, size_t strideX, size_t strideT,
                         size_t padTop, size_t padBottom, size_t padLeft, size_t padRight, size_t padT) ;

  template<> vl::Error
  row2vol<vl::GPU, float>(vl::Context& context,
                         float* data,
                         float const* stacked,
                         size_t height, size_t width, size_t length, size_t depth,
                         size_t windowHeight, size_t windowWidth, size_t windowLength,
                         size_t strideY, size_t strideX, size_t strideT,
                         size_t padTop, size_t padBottom, size_t padLeft, size_t padRight, size_t padT) ;
#endif

} }

#endif /* defined(__vl__vol2row__) */
