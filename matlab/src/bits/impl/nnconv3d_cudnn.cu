// @file nnconv3d_cudnn.cu
// @brief 3D Convolution block CuDNN-based implementation.
// @author Tuan-Hung VU

/*
Copyright (C) 2015-16 Tuan-Hung VU.
All rights reserved.
*/

#if !defined(ENABLE_GPU) | !defined(ENABLE_CUDNN)
#error "nnconv3d_cudnn.hpp cannot be compiled without GPU and CUDNN support."
#endif

#include "nnconv3d_cudnn.hpp"
#include "cudnnhelper.hpp"
#include "../datacu.hpp"
#include <assert.h>
#include <algorithm>

using namespace vl ;

#define CHECK(x) \
{ \
cudnnError = x ; \
if (cudnnError != CUDNN_STATUS_SUCCESS) { \
error = context.setError(context.getCudaHelper().catchCudnnError(cudnnError, \
STRINGIZE(__FILE__) ":" STRINGIZE(__LINE__))) ; \
goto done ; \
} }

/* ---------------------------------------------------------------- */
/*                                             nnconv3d_forward_cudnn */
/* ---------------------------------------------------------------- */

namespace vl { namespace impl {

  template<vl::Type dataType>
  vl::Error
  vl::impl::nnconv3d_cudnn<dataType>::forward(Context& context,
                                            Tensor output, double outputMult,
                                            Tensor data, double dataMult,
                                            Tensor filters,
                                            Tensor biases,
                                            int strideY, int strideX, int strideT,
                                            int padTop, int padBottom,
                                            int padLeft, int padRight, int padT)
  {
    assert(output) ;
    assert(data) ;
    assert(filters) ;

    typedef typename DataTypeTraits<dataType>::type type ;

    cudnnTensorDescriptor_t outputDesc, biasesDesc, dataDesc ;
    cudnnFilterDescriptor_t filtersDesc ;
    cudnnConvolutionDescriptor_t convDesc ;
    bool outputDescInitialized = false ;
    bool biasesDescInitialized = false ;
    bool dataDescInitialized = false ;
    bool filtersDescInitialized = false ;
    bool convDescInitialized = false ;

    void* workSpace = NULL ;

    int numGroups = data.getDimension(3) / filters.getDimension(3) ;
    int numFiltersPerGroup = filters.getDimension(4) / numGroups ;

    if (padLeft != padRight) return vl::vlErrorUnsupported ;
    if (padTop != padBottom) return vl::vlErrorUnsupported ;
    if (filters.getDimension(0) > data.getDimension(0)) return vl::vlErrorUnsupported ;
    if (filters.getDimension(1) > data.getDimension(1)) return vl::vlErrorUnsupported ;

    cudnnStatus_t cudnnError = CUDNN_STATUS_SUCCESS ;
    vl::Error error = vl::vlSuccess ;
    cudnnHandle_t handle ;

    // init dim stride
    int nOutputDims = 5;
    int outputDims[5] = {output.getDimension(4), numFiltersPerGroup, output.getDimension(2), output.getDimension(1), output.getDimension(0)};
    int outputStrides[5] = {output.getDimension(3)*output.getDimension(2)*output.getDimension(1)*output.getDimension(0), output.getDimension(2)*output.getDimension(1)*output.getDimension(0), output.getDimension(1)*output.getDimension(0), output.getDimension(0), 1};

    int nDataDims = 5;
    int dataDims[5] = {data.getDimension(4), data.getDimension(3) / numGroups, data.getDimension(2), data.getDimension(1), data.getDimension(0)};
    int dataStrides[5] = {data.getDimension(3)*data.getDimension(2)*data.getDimension(1)*data.getDimension(0), data.getDimension(2)*data.getDimension(1)*data.getDimension(0), data.getDimension(1)*data.getDimension(0), data.getDimension(0), 1};

    int nFilterDims = 5;
    int filterDims[5] = {numFiltersPerGroup, filters.getDimension(3), filters.getDimension(2), filters.getDimension(1), filters.getDimension(0)};

    int nBiasDims = 5;
    int biasDims[5] = {1,biases.getNumElements() / numGroups,1,1,1};
    int biasStrides[5] = {1,1,1,1,1};

    int convArrayLength = 3;
    int convPad[3] = {padLeft, padTop, padT};
    int convStride[3] = {strideX, strideY, strideT};
    int convUpscale[3] = {1,1,1};

    int tensorOutputDims[5];
    // Get CuDNN
    CHECK(context.getCudaHelper().getCudnnHandle(&handle)) ;

    // Get tensor descripotrs
    CHECK(cudnnCreateTensorDescriptor(&outputDesc)) ;
    outputDescInitialized = true ;
    CHECK(cudnnSetTensorNdDescriptor(outputDesc,
                                       DataTypeToCudnn<dataType>::id ,
                                       nOutputDims,
                                       outputDims,
                                       outputStrides));


    CHECK(cudnnCreateTensorDescriptor(&dataDesc)) ;
    dataDescInitialized = true ;
    CHECK(cudnnSetTensorNdDescriptor(dataDesc,
                                       DataTypeToCudnn<dataType>::id,
                                       nDataDims,
                                       dataDims,
                                       dataStrides)) ;

    CHECK(cudnnCreateFilterDescriptor(&filtersDesc)) ;
    filtersDescInitialized = true ;

    filterDims[0] = numFiltersPerGroup;
    CHECK(cudnnSetFilterNdDescriptor(filtersDesc,
                                     DataTypeToCudnn<dataType>::id,
                                     IF_CUDNN_GE5(CUDNN_TENSOR_NCHW COMMA)
                                     nFilterDims,
                                     filterDims)) ;

    if (biases) {
      CHECK(cudnnCreateTensorDescriptor(&biasesDesc)) ;
      biasesDescInitialized = true ;
      biasDims[1] = biases.getNumElements() / numGroups;
      CHECK(cudnnSetTensorNdDescriptor(biasesDesc,
                                       DataTypeToCudnn<dataType>::id ,
                                       nBiasDims,
                                       biasDims,
                                       biasStrides)) ;
    }

    // Get convolution descriptor
    CHECK(cudnnCreateConvolutionDescriptor(&convDesc)) ;
    convDescInitialized = true ;
    CHECK(cudnnSetConvolutionNdDescriptor(convDesc,
                                          convArrayLength,
                                          convPad,
                                          convStride,
                                          convUpscale,
                                          CUDNN_CROSS_CORRELATION,
                                          DataTypeToCudnn<dataType>::id)) ;

    // Sanity check
#if 1
    {
      cudnnGetConvolutionNdForwardOutputDim(convDesc,
                                            dataDesc,
                                            filtersDesc,
                                            5,
                                            tensorOutputDims) ;
      bool sane =
      output.getDimension(0) == tensorOutputDims[0] &&
      numFiltersPerGroup == tensorOutputDims[1] &&
      output.getDimension(2) == tensorOutputDims[2] &&
      output.getDimension(3) == tensorOutputDims[3] &&
      output.getDimension(4) == tensorOutputDims[4];
      assert(sane) ;
    }
#endif

    context.getCudaHelper().cudnnConvolutionFwdWorkSpaceUsed = 0 ;
    context.getCudaHelper().cudnnConvolutionBwdFilterWorkSpaceUsed = 0 ;
    context.getCudaHelper().cudnnConvolutionBwdDataWorkSpaceUsed = 0 ;

    if (!context.getCudaHelper().cudnnConvolutionFwdSpecificAlgo) {
      // Determine algorithm automatically
      CHECK(cudnnGetConvolutionForwardAlgorithm(handle,
                                                dataDesc,
                                                filtersDesc,
                                                convDesc,
                                                outputDesc,
                                                context.getCudaHelper().cudnnConvolutionFwdPreference,
                                                context.getCudaHelper().cudnnConvolutionFwdWorkSpaceLimit,
                                                &context.getCudaHelper().cudnnConvolutionFwdAlgo)) ;
    }

    // Get workspace size
    CHECK(cudnnGetConvolutionForwardWorkspaceSize(handle,
                                                  dataDesc,
                                                  filtersDesc,
                                                  convDesc,
                                                  outputDesc,
                                                  context.getCudaHelper().cudnnConvolutionFwdAlgo,
                                                  &context.getCudaHelper().cudnnConvolutionFwdWorkSpaceUsed)) ;

    // Get workspace
    if (context.getCudaHelper().cudnnConvolutionFwdWorkSpaceUsed > 0) {
      workSpace = context.getWorkspace(vl::GPU, context.getCudaHelper().cudnnConvolutionFwdWorkSpaceUsed) ;
      if (workSpace == NULL) {
        error = context.getLastError() ;
        goto done ;
      }
    }

    // Perform convolution for each filter group
    for (int g = 0  ; g < numGroups ; ++g) {
      ptrdiff_t dataGrpOffset = (data.getDimension(0) * data.getDimension(1) * data.getDimension(2) * filters.getDimension(3)) *  g ;
      ptrdiff_t filtersGrpOffset = (filters.getDimension(0) * filters.getDimension(1) * filters.getDimension(3)) * numFiltersPerGroup * g ;
      ptrdiff_t outputGrpOffset = (output.getDimension(0) * output.getDimension(1) * numFiltersPerGroup) * g ;
      ptrdiff_t biasesGrpOffset = numFiltersPerGroup * g ;

      type alpha = dataMult ;
      type beta = outputMult ;
      CHECK(cudnnConvolutionForward(handle,
                                    &alpha,
                                    dataDesc, (type const*)data.getMemory() + dataGrpOffset,
                                    filtersDesc, (type const*)filters.getMemory() + filtersGrpOffset,
                                    convDesc,
                                    context.getCudaHelper().cudnnConvolutionFwdAlgo,
                                    workSpace, context.getCudaHelper().cudnnConvolutionFwdWorkSpaceUsed,
                                    &beta,
                                    outputDesc, (type*)output.getMemory() + outputGrpOffset)) ;

      if (biases) {
        type alpha = 1.0f ;
        type beta = 1.0f ;
#if (CUDNN_VERSION < 4000)
        CHECK(cudnnAddTensor(handle,
                             CUDNN_ADD_SAME_C,
                             &alpha,
                             biasesDesc, (type const*)biases.getMemory() + biasesGrpOffset,
                             &beta,
                             outputDesc, (type*)output.getMemory() + outputGrpOffset)) ;
#else
        CHECK(cudnnAddTensor(handle,
                             &alpha,
                             biasesDesc, (type const*)biases.getMemory() + biasesGrpOffset,
                             &beta,
                             outputDesc, (type*)output.getMemory() + outputGrpOffset)) ;
#endif
      }
    }

    /* cleanup */
  done:
    if (convDescInitialized) { cudnnDestroyConvolutionDescriptor(convDesc) ; }
    if (filtersDescInitialized) { cudnnDestroyFilterDescriptor(filtersDesc) ; }
    if (dataDescInitialized) { cudnnDestroyTensorDescriptor(dataDesc) ; }
    if (biasesDescInitialized) { cudnnDestroyTensorDescriptor(biasesDesc) ; }
    if (outputDescInitialized) { cudnnDestroyTensorDescriptor(outputDesc) ; }
    return context.passError(error, __func__) ;
  }

  /* ---------------------------------------------------------------- */
  /*                                            nnconv3d_backward_cudnn */
  /* ---------------------------------------------------------------- */

  template<vl::Type dataType>
  vl::Error
  vl::impl::nnconv3d_cudnn<dataType>::backward(Context& context,
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
    typedef typename DataTypeTraits<dataType>::type type ;

    /* no derDataDesc needed as same as dataDesc */
    cudnnTensorDescriptor_t dataDesc, derBiasesDesc, derOutputDesc ;
    cudnnFilterDescriptor_t filtersDesc ;
    cudnnConvolutionDescriptor_t convDesc ;
    bool dataDescInitialized = false ;
    bool derBiasesDescInitialized = false ;
    bool derOutputDescInitialized = false ;
    bool filtersDescInitialized = false ;
    bool convDescInitialized = false ;

#if (CUDNN_VERSION >= 3000)
    void* workSpace = NULL ;
    size_t workSpaceSize = 0 ;
#endif

    ptrdiff_t numGroups = 1 ;
    ptrdiff_t numFiltersPerGroup = 0 ;
    ptrdiff_t filtersVolume = 0 ;

    if (padLeft != padRight) return vl::vlErrorUnsupported ;
    if (padTop != padBottom) return vl::vlErrorUnsupported ;

    cudnnStatus_t cudnnError = CUDNN_STATUS_SUCCESS ;
    vl::Error error = vl::vlSuccess ;
    cudnnHandle_t handle ;

    // init dim and strides
    int nDataDims = 5;
    int dataDims[5] = {data.getDimension(4), data.getDimension(3) / numGroups, data.getDimension(2), data.getDimension(1), data.getDimension(0)};
    int dataStrides[5] = {data.getDimension(3)*data.getDimension(2)*data.getDimension(1)*data.getDimension(0), data.getDimension(2)*data.getDimension(1)*data.getDimension(0), data.getDimension(1)*data.getDimension(0), data.getDimension(0), 1}; 
   
    int nFilterDims = 5;
    int filterDims[5] = {numFiltersPerGroup, filters.getDimension(3), filters.getDimension(2), filters.getDimension(1), filters.getDimension(0)};

    int nDerFilterDims = 5;
    int derFilterDims[5] = {numFiltersPerGroup, derFilters.getDimension(3), derFilters.getDimension(2), derFilters.getDimension(1), derFilters.getDimension(0)};

    int convArrayLength = 3;
    int convPad[3] = {padLeft, padTop, padT};
    int convStride[3] = {strideX, strideY, strideT};
    int convUpscale[3] = {1,1,1};

    int nDerOutputDims = 5;
    int derOutputDims[5] = {derOutput.getDimension(4), numFiltersPerGroup, derOutput.getDimension(2), derOutput.getDimension(1), derOutput.getDimension(0)};
    int derOutputStrides[5] = {derOutput.getDimension(3)*derOutput.getDimension(2)*derOutput.getDimension(1)*derOutput.getDimension(0), derOutput.getDimension(2)*derOutput.getDimension(1)*derOutput.getDimension(0), derOutput.getDimension(1)*derOutput.getDimension(0), derOutput.getDimension(0), 1};
 
    int nDerBiasDims = 5;
    int derBiasDims[5] = {1,derBiases.getNumElements() / numGroups,1,1,1};
    int derBiasStrides[5] = {1,1,1,1,1};

    // Get CuDNN
    CHECK(context.getCudaHelper().getCudnnHandle(&handle)) ;

    // Get the dimensions of the tensrors involved
    // If derData is specified (hence comptued as output), use this
    // tensor as a basis to compute such dimensions, otherwise use derFilters.

    if (derData) {
      assert(filters) ;
      numGroups = derData.getDimension(3) / filters.getDimension(3) ;
      numFiltersPerGroup = filters.getDimension(4) / numGroups ;
      filtersVolume = filters.getDimension(0) * filters.getDimension(1) * filters.getDimension(2) * filters.getDimension(3) ;

      CHECK(cudnnCreateTensorDescriptor(&dataDesc)) ;
      dataDescInitialized = true ;
      dataDims[1] = data.getDimension(3) / numGroups;
      CHECK(cudnnSetTensorNdDescriptor(dataDesc,
                                         DataTypeToCudnn<dataType>::id ,
                                         nDataDims,
                                         dataDims,
                                         dataStrides)) ;


      CHECK(cudnnCreateFilterDescriptor(&filtersDesc)) ;
      filtersDescInitialized = true ;
      filterDims[0] = numFiltersPerGroup;
      CHECK(cudnnSetFilterNdDescriptor(filtersDesc,
                                       DataTypeToCudnn<dataType>::id ,
                                       IF_CUDNN_GE5(CUDNN_TENSOR_NCHW COMMA)
                                       nFilterDims,
                                       filterDims)) ;

    } else if (derFilters) {
      assert(data) ;
      numGroups = data.getDimension(3) / derFilters.getDimension(3) ;
      numFiltersPerGroup = derFilters.getDimension(4) / numGroups ;
      filtersVolume = derFilters.getDimension(0) * derFilters.getDimension(1) * derFilters.getDimension(2) * derFilters.getDimension(3);

      CHECK(cudnnCreateTensorDescriptor(&dataDesc)) ;
      dataDescInitialized = true ;
      dataDims[1] = data.getDimension(3) / numGroups;
      CHECK(cudnnSetTensorNdDescriptor(dataDesc,
                                         DataTypeToCudnn<dataType>::id ,
                                         nDataDims,
                                         dataDims,
                                         dataStrides)) ;

      CHECK(cudnnCreateFilterDescriptor(&filtersDesc)) ;
      filtersDescInitialized = true ;
      derFilterDims[0] = numFiltersPerGroup;
      CHECK(cudnnSetFilterNdDescriptor(filtersDesc,
                                       DataTypeToCudnn<dataType>::id ,
                                       IF_CUDNN_GE5(CUDNN_TENSOR_NCHW COMMA)
                                       nDerFilterDims,
                                       derFilterDims)) ;

    }

    CHECK(cudnnCreateConvolutionDescriptor(&convDesc)) ;
    convDescInitialized = true ;

    CHECK(cudnnSetConvolutionNdDescriptor(convDesc,
                                          convArrayLength,
                                          convPad,
                                          convStride,
                                          convUpscale,
                                          CUDNN_CROSS_CORRELATION,
                                          DataTypeToCudnn<dataType>::id)) ;

    // Must have derOutput for all derivatives
    assert(derOutput) ;
    CHECK(cudnnCreateTensorDescriptor(&derOutputDesc)) ;
    derOutputDescInitialized = true ;

    derOutputDims[1] = numFiltersPerGroup;
    CHECK(cudnnSetTensorNdDescriptor(derOutputDesc,
                                       DataTypeToCudnn<dataType>::id ,
                                       nDerOutputDims, 
                                       derOutputDims,
                                       derOutputStrides)) ;

    // for derivatives w.r.t. bias
    if (derBiases) {
      CHECK(cudnnCreateTensorDescriptor(&derBiasesDesc)) ;
      derBiasesDescInitialized = true ;

      derBiasDims[1] = derBiases.getNumElements() / numGroups;
      CHECK(cudnnSetTensorNdDescriptor(derBiasesDesc, //CUDNN_TENSOR_NCHW,
                                       DataTypeToCudnn<dataType>::id ,
                                       nDerBiasDims,
                                       derBiasDims,
                                       derBiasStrides)) ;

    }


    context.getCudaHelper().cudnnConvolutionFwdWorkSpaceUsed = 0 ;
    context.getCudaHelper().cudnnConvolutionBwdFilterWorkSpaceUsed = 0 ;
    context.getCudaHelper().cudnnConvolutionBwdDataWorkSpaceUsed = 0 ;

#if (CUDNN_VERSION >= 3000)

    if (derFilters) {
      // Get filter derivatives algorithm
      CHECK(cudnnGetConvolutionBackwardFilterAlgorithm
            (handle,
             dataDesc,
             derOutputDesc,
             convDesc,
             filtersDesc,
             context.getCudaHelper().cudnnConvolutionBwdFilterPreference,
             context.getCudaHelper().cudnnConvolutionBwdFilterWorkSpaceLimit,
             &context.getCudaHelper().cudnnConvolutionBwdFilterAlgo)) ;

      // Get workspace size
      CHECK(cudnnGetConvolutionBackwardFilterWorkspaceSize
            (handle,
             dataDesc,
             derOutputDesc,
             convDesc,
             filtersDesc,
             context.getCudaHelper().cudnnConvolutionBwdFilterAlgo,
             &context.getCudaHelper().cudnnConvolutionBwdFilterWorkSpaceUsed)) ;
      workSpaceSize = std::max(workSpaceSize, context.getCudaHelper().cudnnConvolutionBwdFilterWorkSpaceUsed) ;
    }

    if (derData) {
      // Get data derivatives
      CHECK(cudnnGetConvolutionBackwardDataAlgorithm
            (handle,
             filtersDesc,
             derOutputDesc,
             convDesc,
             dataDesc,
             context.getCudaHelper().cudnnConvolutionBwdDataPreference,
             context.getCudaHelper().cudnnConvolutionBwdDataWorkSpaceLimit,
             &context.getCudaHelper().cudnnConvolutionBwdDataAlgo)) ;

      // Get workspace size
      CHECK(cudnnGetConvolutionBackwardDataWorkspaceSize
            (handle,
             filtersDesc,
             derOutputDesc,
             convDesc,
             dataDesc,
             context.getCudaHelper().cudnnConvolutionBwdDataAlgo,
             &context.getCudaHelper().cudnnConvolutionBwdDataWorkSpaceUsed)) ;
      workSpaceSize = std::max(workSpaceSize, context.getCudaHelper().cudnnConvolutionBwdDataWorkSpaceUsed) ;
    }

    // Get workspace
    if (workSpaceSize > 0) {
      workSpace = context.getWorkspace(vl::GPU, workSpaceSize) ;
      if (workSpace == NULL) {
        error = context.getLastError() ;
        goto done ;
      }
    }
#endif

    // Perform backward convolution for each filter group
    for (int g = 0  ; g < numGroups ; ++g) {
      ptrdiff_t filtersGrpOffset = filtersVolume * numFiltersPerGroup  * g ;
      ptrdiff_t derOutputGrpOffset = (derOutput.getDimension(0) * derOutput.getDimension(1) * derOutput.getDimension(2) * numFiltersPerGroup) * g ;

      if (derBiases) {
        ptrdiff_t derBiasesGrpOffset = numFiltersPerGroup * g ;
        type alpha = 1 ;
        type beta = 0 ;
        CHECK(cudnnConvolutionBackwardBias
              (handle,
               &alpha,
               derOutputDesc, (type const*)derOutput.getMemory() + derOutputGrpOffset,
               &beta,
               derBiasesDesc, (type*)derBiases.getMemory() + derBiasesGrpOffset)) ;
      }

      if (derFilters) {
        ptrdiff_t dataGrpOffset = (data.getHeight() * data.getWidth() * derFilters.getDepth()) *  g ;
        type alpha = 1 ;
        type beta = 0 ;
#if (CUDNN_VERSION >= 3000)
        CHECK(
              IF_CUDNN_GE4(cudnnConvolutionBackwardFilter)
              IF_CUDNN_GE3_LT4(cudnnConvolutionBackwardFilter_v3)
              (handle,
               &alpha,
               dataDesc, (type const*)data.getMemory() + dataGrpOffset,
               derOutputDesc, (type const*)derOutput.getMemory() + derOutputGrpOffset,
               convDesc,
               context.getCudaHelper().cudnnConvolutionBwdFilterAlgo,
               workSpace, workSpaceSize,
               &beta,
               filtersDesc, (type*)derFilters.getMemory() + filtersGrpOffset)) ;
#else
        CHECK(cudnnConvolutionBackwardFilter
              (handle,
               &alpha,
               dataDesc, (type const*)data.getMemory() + dataGrpOffset,
               derOutputDesc, (type const*)derOutput.getMemory() + derOutputGrpOffset,
               convDesc,
               &beta,
               filtersDesc, (type*)derFilters.getMemory() + filtersGrpOffset)) ;
#endif
      }

      if (derData) {
        ptrdiff_t dataGrpOffset = (derData.getDimension(0) * derData.getDimension(1) * derData.getDimension(2) * filters.getDimension(3)) *  g ;
        type alpha = 1 ;
        type beta = 0 ;

#if (CUDNN_VERSION >= 3000)
        CHECK(
              IF_CUDNN_GE4(cudnnConvolutionBackwardData)
              IF_CUDNN_GE3_LT4(cudnnConvolutionBackwardData_v3)
              (handle,
               &alpha,
               filtersDesc, (type const*)filters.getMemory() + filtersGrpOffset,
               derOutputDesc, (type const*)derOutput.getMemory() + derOutputGrpOffset,
               convDesc,
               context.getCudaHelper().cudnnConvolutionBwdDataAlgo,
               workSpace, workSpaceSize,
               &beta,
               dataDesc, (type*)derData.getMemory() + dataGrpOffset)) ;
#else
        CHECK(cudnnConvolutionBackwardData
              (handle,
               &alpha,
               filtersDesc, filters.getMemory() + filtersGrpOffset,
               derOutputDesc, derOutput.getMemory() + derOutputGrpOffset,
               convDesc,
               &beta,
               dataDesc, derData.getMemory() + dataGrpOffset)) ;
#endif
      }
    }

  done:
    if (convDescInitialized) { cudnnDestroyConvolutionDescriptor(convDesc) ; }
    if (filtersDescInitialized) { cudnnDestroyFilterDescriptor(filtersDesc) ; }
    if (derOutputDescInitialized) { cudnnDestroyTensorDescriptor(derOutputDesc) ; }
    if (derBiasesDescInitialized) { cudnnDestroyTensorDescriptor(derBiasesDesc) ; }
    if (dataDescInitialized) { cudnnDestroyTensorDescriptor(dataDesc) ; }
    return context.passError(error, __func__) ;
  }

} }

// Instantiations
template struct vl::impl::nnconv3d_cudnn<vl::vlTypeFloat> ;

#ifdef ENABLE_DOUBLE
template struct vl::impl::nnconv3d_cudnn<vl::vlTypeDouble> ;
#endif



