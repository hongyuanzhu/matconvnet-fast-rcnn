// @file vl_nnroipooling.cu
// @brief Porting roi data layer from fast-rcnn
// @author Tuan-Hung Vu

/*
Copyright (C) 2015-2016 Tuan-Hung Vu.
All rights reserved.

This file is made available under the terms of the BSD license.
*/

#include "bits/mexutils.h"
#include "bits/datamex.hpp"
#include "bits/nnroipooling.hpp"

#if ENABLE_GPU
#include "bits/datacu.hpp"
#endif

#include <assert.h>

/* option codes */
enum {
  opt_method = 0,
  opt_spatial_scale,
  opt_verbose,
  opt_cudnn,
  opt_no_cudnn,
} ;

/* options */
vlmxOption  options [] = {
  {"Method",           1,   opt_method            },
  {"SpatialScale",     1,   opt_spatial_scale     },
  {"Verbose",          0,   opt_verbose           },
  {"CUDNN",            0,   opt_cudnn             },
  {"NoCUDNN",          0,   opt_no_cudnn          },
  {0,                  0,   0                     }
} ;

/* ---------------------------------------------------------------- */
/*                                                          Context */
/* ---------------------------------------------------------------- */

vl::MexContext context ;

/*
 Resetting the context here resolves a crash when MATLAB quits and
 the ~Context function is implicitly called on unloading the MEX file.
 */
void atExit()
{
  context.clear() ;
}

/* ---------------------------------------------------------------- */
/*                                                       MEX driver */
/* ---------------------------------------------------------------- */

enum {
  IN_DATA = 0, IN_SIZE, IN_ROI, IN_DEROUTPUT, IN_ARGMAX, IN_END
} ;

enum {
  OUT_RESULT = 0, OUT_ARGMAX, OUT_END
} ;

void mexFunction(int nout, mxArray *out[],
                 int nin, mxArray const *in[])
{
  int poolWidth ;
  int poolHeight ;
  float spatial_scale;
  vl::PoolingMethod method = vl::vlPoolingMax ;
  bool backMode = false ;

  int verbosity = 0 ;
  int opt ;
  int next = IN_END ;
  mxArray const *optarg ;

  /* -------------------------------------------------------------- */
  /*                                            Check the arguments */
  /* -------------------------------------------------------------- */

  mexAtExit(atExit) ;

  if (nin < 3) {
    mexErrMsgTxt("The arguments are less than three.") ;
  }

  if (nin > 3 && vlmxIsString(in[3],-1)) {
    next = 3 ;
    backMode = 0 ;
  } else {
    backMode = (nin >= 5) ;
  }

  while ((opt = vlmxNextOption (in, nin, options, &next, &optarg)) >= 0) {
    switch (opt) {
      case opt_spatial_scale:
        spatial_scale = (float)mxGetPr(optarg)[0] ;
        break;

      case opt_verbose :
        ++ verbosity ;
        break ;

      case opt_method :
        if (!vlmxIsString(optarg,-1)) {
           vlmxError(vlmxErrInvalidArgument, "METHOD is not a string.") ;
        }
        if (vlmxIsEqualToStringI(optarg, "max")) {
          method = vl::vlPoolingMax ;
        } else {
          vlmxError(vlmxErrInvalidArgument, "METHOD is not a supported method.") ;
        }
        break;

      case opt_no_cudnn :
#if ENABLE_CUDNN
        context.getCudaHelper().setCudnnEnabled(false) ;
#endif
        break ;

      case opt_cudnn :
#if ENABLE_CUDNN
        context.getCudaHelper().setCudnnEnabled(true) ;
#endif
        break ;

      default:
        break ;
    }
  }

  vl::MexTensor data(context) ;
  vl::MexTensor derOutput(context) ;
  vl::MexTensor argmax(context) ;

  data.init(in[IN_DATA]) ;
  data.reshape(4) ;

  // ROIs
  vl::MexTensor rois(context) ;			// TODO: create a better data type, rather than 4D MexTensor, to store ROIs
  rois.init(in[IN_ROI]) ;

  if (backMode) { 
    derOutput.init(in[IN_DEROUTPUT]) ; 
    argmax.init(in[IN_ARGMAX]) ;
  }

  if (backMode && ! vl::areCompatible(data, derOutput)) {
    mexErrMsgTxt("DATA and DEROUTPUT are not both CPU or GPU arrays.") ;
  }

  if (!vlmxIsPlainMatrix(in[IN_SIZE],-1,-1)) {
    mexErrMsgTxt("SIZE is not a plain matrix.") ;
  }
  switch (mxGetNumberOfElements(in[IN_SIZE])) {
    case 1:
      poolHeight = mxGetPr(in[IN_SIZE])[0] ;
      poolWidth = poolHeight ;
      break ;
    case 2:
      poolHeight = mxGetPr(in[IN_SIZE])[0] ;
      poolWidth = mxGetPr(in[IN_SIZE])[1] ;
      break ;
    default:
      mexErrMsgTxt("SIZE has neither one nor two elements.") ;
  }
  
  vl::TensorShape outputGeom(1,
                                1,
                                poolHeight*poolWidth*data.getDepth(),
                                rois.getWidth()) ;

  /* Create output buffers */
  vl::Device deviceType = data.getDeviceType() ;
  vl::Type dataType = data.getDataType() ;

  vl::MexTensor output(context) ;
  vl::MexTensor derData(context) ;
  vl::MexTensor derFilters(context) ;
  vl::MexTensor derBiases(context) ;

  if (!backMode) {
    output.init(deviceType, dataType, outputGeom) ;
    argmax.init(deviceType, dataType, outputGeom) ;
  } else {
    derData.init(deviceType, dataType, data.getShape()) ;
  }

  if (verbosity > 0) {
    mexPrintf("vl_nnroipooling: %s; %s", backMode?"backward":"forward", (data.getDeviceType()==vl::GPU) ? "GPU" : "CPU") ;
    if (data.getDeviceType() == vl::GPU) {
#if ENABLE_CUDNN
      mexPrintf("; %s\n", context.getCudaHelper().getCudnnEnabled() ? "cuDNN" : "MatConvNet") ;
#else
      mexPrintf("; MatConvNet\n") ;
#endif
    } else {
      mexPrintf("; MatConvNet\n") ;
    }
    vl::print("vl_nnroipooling: data: ", data) ;
    vl::print("vl_nnroipooling: rois: ", rois) ;
    mexPrintf("vl_nnroipooling: roi pooling: %d x %d\n", poolHeight, poolWidth);
    mexPrintf("vl_nnroipooling: spatial scale: %f\n", spatial_scale);
    mexPrintf("vl_nnroipooling: method: %s\n", (method == vl::vlPoolingMax) ? "max" : "avg") ;
    if (backMode) {
      vl::print("vl_nnroipooling: derOutput: ", derOutput) ;
      vl::print("vl_nnroipooling: derData: ", derData) ;
    } else {
      vl::print("vl_nnroipooling: output: ", output) ;
      vl::print("vl_nnroipooling: argmax: ", argmax) ;
    }
  }

  /* -------------------------------------------------------------- */
  /*                                                    Do the work */
  /* -------------------------------------------------------------- */

  vl::Error error ;
  if (!backMode) {
    error = vl::nnroipooling_forward(context,
                                  output, argmax, data, rois,
                                  method,
                                  poolHeight, poolWidth,
                                  spatial_scale) ;
  } else {
    error = vl::nnroipooling_backward(context,
                                   derData, data, rois, derOutput, argmax,
                                   method,
                                   poolHeight, poolWidth,
                                   spatial_scale) ;
  }

  /* -------------------------------------------------------------- */
  /*                                                         Finish */
  /* -------------------------------------------------------------- */

  if (error != vl::vlSuccess) {
    mexErrMsgTxt(context.getLastErrorMessage().c_str()) ;
  }
  if (backMode) {
    out[OUT_RESULT] = derData.relinquish() ;
  } else {
    out[OUT_RESULT] = output.relinquish() ;
    out[OUT_ARGMAX] = argmax.relinquish() ;
  }
}