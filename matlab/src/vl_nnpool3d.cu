// @file vl_nnpool3d.cu
// @brief 3D Pooling block MEX wrapper
// @author Tuan-Hung VU

/*
Copyright (C) 2016 Tuan-Hung VU.
All rights reserved.
*/

#include "bits/mexutils.h"
#include "bits/datamex.hpp"
#include "bits/nnpooling3d.hpp"

#if ENABLE_GPU
#include "bits/datacu.hpp"
#endif

#include <assert.h>

/* option codes */
enum {
  opt_stride = 0,
  opt_pad,
  opt_method,
  opt_verbose,
  opt_cudnn,
  opt_no_cudnn,
} ;

/* options */
vlmxOption  options [] = {
  {"Stride",           1,   opt_stride            },
  {"Pad",              1,   opt_pad               },
  {"Method",           1,   opt_method            },
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
  IN_DATA = 0, IN_SIZE, IN_DEROUTPUT, IN_END
} ;

enum {
  OUT_RESULT = 0, OUT_END
} ;

void mexFunction(int nout, mxArray *out[],
                 int nin, mxArray const *in[])
{
  int poolWidth ;
  int poolHeight ;
  int poolLength ;
  int strideX = 1 ;
  int strideY = 1 ;
  int strideL = 1;
  int padLeft = 0 ;
  int padRight = 0 ;
  int padTop = 0 ;
  int padBottom = 0 ;
  int padLength = 0 ;
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

  if (nin < 2) {
    mexErrMsgTxt("The arguments are less than two.") ;
  }

  if (nin > 2 && vlmxIsString(in[2],-1)) {
    next = 2 ;
    backMode = 0 ;
  } else {
    backMode = (nin >= 3) ;
  }

  while ((opt = vlmxNextOption (in, nin, options, &next, &optarg)) >= 0) {
    switch (opt) {
      case opt_verbose :
        ++ verbosity ;
        break ;

      case opt_stride :
        if (!vlmxIsPlainMatrix(optarg,-1,-1)) {
          mexErrMsgTxt("STRIDE is not a plain matrix.") ;
        }
        switch (mxGetNumberOfElements(optarg)) {
          case 1:
            strideY = (int)mxGetPr(optarg)[0] ;
            strideX = strideY ;
            strideL = strideY ;
            break ;
          case 3:
            strideY = (int)mxGetPr(optarg)[0] ;
            strideX = (int)mxGetPr(optarg)[1] ;
            strideL = (int)mxGetPr(optarg)[2] ;
            break ;
          default:
            mexErrMsgTxt("STRIDE has neither one nor two elements.") ;
        }
        break ;

      case opt_pad :
        if (!vlmxIsPlainMatrix(optarg,-1,-1)) {
          mexErrMsgTxt("PAD is not a plain matrix.") ;
        }
        switch (mxGetNumberOfElements(optarg)) {
          case 1:
            padLeft = (int)mxGetPr(optarg)[0] ;
            padRight = padLeft ;
            padTop = padLeft ;
            padBottom = padLeft ;
            padLength = padLeft ;
            break ;
          case 5:
            padTop = (int)mxGetPr(optarg)[0] ;
            padBottom = (int)mxGetPr(optarg)[1] ;
            padLeft = (int)mxGetPr(optarg)[2] ;
            padRight = (int)mxGetPr(optarg)[3] ;
            padLength = (int)mxGetPr(optarg)[4] ;
            break ;
          default:
            mexErrMsgTxt("PAD has neither one nor five elements.") ;
        }
        break;

      case opt_method :
        if (!vlmxIsString(optarg,-1)) {
           vlmxError(vlmxErrInvalidArgument, "METHOD is not a string.") ;
        }
        if (vlmxIsEqualToStringI(optarg, "max")) {
          method = vl::vlPoolingMax ;
        } else if (vlmxIsEqualToStringI(optarg, "avg")) {
          method = vl::vlPoolingAverage ;
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

  data.init(in[IN_DATA]) ;
  data.reshape(5) ; // -> 5 dimensions

  if (backMode) {
    derOutput.init(in[IN_DEROUTPUT]) ;
    derOutput.reshape(5) ; // -> 5 dimensions
  }

  if (backMode && ! vl::areCompatible(data, derOutput)) {
    mexErrMsgTxt("DATA and DEROUTPUT do not have compatible formats.") ;
  }

  if (!vlmxIsPlainMatrix(in[IN_SIZE],-1,-1)) {
    mexErrMsgTxt("SIZE is not a plain matrix.") ;
  }
  switch (mxGetNumberOfElements(in[IN_SIZE])) {
    case 1:
      poolHeight = mxGetPr(in[IN_SIZE])[0] ;
      poolWidth = poolHeight ;
      poolLength = poolHeight ;
      break ;
    case 3:
      poolHeight = mxGetPr(in[IN_SIZE])[0] ;
      poolWidth = mxGetPr(in[IN_SIZE])[1] ;
      poolLength = mxGetPr(in[IN_SIZE])[2] ;
      break ;
    default:
      mexErrMsgTxt("SIZE has neither one nor two elements.") ;
  }

  /* Basic compatibility of Shape */
  if (strideX < 1 || strideY < 1 || strideL < 1) {
    mexErrMsgTxt("At least one element of STRIDE is smaller than one.") ;
  }
  if (poolHeight == 0 || poolWidth == 0 || poolLength == 0) {
    mexErrMsgTxt("A dimension of the pooling SIZE is void.") ;
  }
  if (data.getDimension(0) + (padTop+padBottom) < poolHeight ||
      data.getDimension(1) + (padLeft+padRight) < poolWidth ||
      data.getDimension(2) + (padLength+padLength) < poolLength) {
    mexErrMsgTxt("The pooling cube is larger than the DATA (including padding).") ;
  }
  if (padLeft < 0 ||
      padRight < 0 ||
      padTop < 0 ||
      padBottom < 0 ||
      padLength < 0) {
    mexErrMsgTxt("An element of PAD is negative.") ;
  }
  if (padLeft >= poolWidth ||
      padRight >= poolWidth ||
      padTop >= poolHeight  ||
      padBottom >= poolHeight ||
      padLength >= poolLength) {
    mexErrMsgTxt("A padding value is larger or equal to the size of the pooling window.") ;
  }

  /* Get the output Shape */
  vl::TensorShape outputShape;
  outputShape.setDimension(0, (data.getDimension(0) + (padTop+padBottom) - poolHeight)/strideY + 1);
  outputShape.setDimension(1, (data.getDimension(1)  + (padLeft+padRight) - poolWidth)/strideX + 1);
  outputShape.setDimension(2, (data.getDimension(2)  + (padLength+padLength) - poolLength)/strideL + 1);
  outputShape.setDimension(3, data.getDimension(3));
  outputShape.setDimension(4, data.getDimension(4));

  if (backMode && (derOutput != outputShape)) {
    mexErrMsgTxt("DEROUTPUT dimensions are incompatible with X and POOL.") ;
  }

  /* Create output buffers */
  vl::Device deviceType = data.getDeviceType() ;
  vl::Type dataType = data.getDataType() ;
  vl::MexTensor output(context) ;
  vl::MexTensor derData(context) ;

  if (!backMode) {
    output.initWithZeros(deviceType, dataType, outputShape) ;
  } else {
    derData.initWithZeros(deviceType, dataType, data.getShape()) ;
  }

  if (verbosity > 0) {
    mexPrintf("vl_nnpool3d: %s; %s", backMode?"backward":"forward", (data.getDeviceType()==vl::GPU) ? "GPU" : "CPU") ;
    if (data.getDeviceType() == vl::GPU) {
#if ENABLE_CUDNN
      mexPrintf("; %s\n", context.getCudaHelper().getCudnnEnabled() ? "cuDNN" : "MatConvNet") ;
#else
      mexPrintf("; MatConvNet\n") ;
#endif
    } else {
      mexPrintf("; MatConvNet\n") ;
    }
    mexPrintf("vl_nnpool3d: stride: [%d %d %d], pad: [%d %d %d %d %d]\n",
              strideY, strideX, strideL,
              padTop, padBottom, padLeft, padRight, padLength) ;
    vl::print("vl_nnpool3d: data: ", data) ;
    mexPrintf("vl_nnpool3d: pooling: %d x %d x %d\n", poolHeight, poolWidth, poolLength);
    mexPrintf("vl_nnpool3d: method: %s\n", (method == vl::vlPoolingMax) ? "max" : "avg") ;
    if (backMode) {
      vl::print("vl_nnpool3d: derOutput: ", derOutput) ;
      vl::print("vl_nnpool3d: derData: ", derData) ;
    } else {
      vl::print("vl_nnpool3d: output: ", output) ;
    }
  }

  /* -------------------------------------------------------------- */
  /*                                                    Do the work */
  /* -------------------------------------------------------------- */

  vl::Error error ;
  if (!backMode) {
    error = vl::nnpooling3d_forward(context,
                                  output, data,
                                  method,
                                  poolHeight, poolWidth, poolLength,
                                  strideY, strideX, strideL,
                                  padTop, padBottom, padLeft, padRight, padLength) ;
  } else {
    error = vl::nnpooling3d_backward(context,
                                   derData, data, derOutput,
                                   method,
                                   poolHeight, poolWidth, poolLength,
                                   strideY, strideX, strideL,
                                   padTop, padBottom, padLeft, padRight, padLength) ;
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
  }
}
