classdef Conv3D < dagnn.Filter
  properties
    size = [0 0 0 0 0]
    hasBias = true
    %opts = {'cuDNN'} % not yet supported
    opts = {'cuDNN'};
  end

  methods
    function outputs = forward(obj, inputs, params)
      if ~obj.hasBias, params{2} = [] ; end
      outputs{1} = vl_nnconv3d(...
        inputs{1}, params{1}, params{2}, ...
        'pad', obj.pad, ...
        'stride', obj.stride, ...
        obj.opts{:}) ;
    end

    function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
      if ~obj.hasBias, params{2} = [] ; end
      [derInputs{1}, derParams{1}, derParams{2}] = vl_nnconv3d(...
        inputs{1}, params{1}, params{2}, derOutputs{1}, ...
        'pad', obj.pad, ...
        'stride', obj.stride, ...
        obj.opts{:}) ;
    end

    function kernelSize = getKernelSize(obj)
      kernelSize = obj.size(1:3) ;
    end

    function outputSizes = getOutputSizes(obj, inputSizes)
      ks = obj.getKernelSize() ;
      outputSizes{1} = [...
        fix((inputSizes{1}(1) + obj.pad(1) + obj.pad(2) - ks(1)) / obj.stride(1)) + 1, ...
        fix((inputSizes{1}(2) + obj.pad(3) + obj.pad(4) - ks(2)) / obj.stride(2)) + 1, ...
        fix((inputSizes{1}(3) + obj.pad(5) + obj.pad(5) - ks(3)) / obj.stride(3)) + 1, ...
        1, ...
        inputSizes{1}(4)] ;
      %outputSizes = getOutputSizes@dagnn.Filter(obj, inputSizes) ;
      outputSizes{1}(4) = obj.size(5) ;
    end

    function params = initParams(obj)
      sc = sqrt(2 / prod(obj.size(1:4))) ;
      params{1} = randn(obj.size,'single') * sc ;
      if obj.hasBias
        params{2} = zeros(obj.size(5),1,'single') * sc ;
      end
    end

    function set.size(obj, ksize)
      % make sure that ksize has 4 dimensions
      ksize = [ksize(:)' 1 1 1 1 1] ;
      obj.size = ksize(1:5) ;
    end

    function obj = Conv3D(varargin)
      obj.load(varargin) ;
      % normalize field by implicitly calling setters defined in
      % dagnn.Filter and here
      obj.size = obj.size ;
      obj.stride = obj.stride ;
      obj.pad = obj.pad ;
    end
  end
end
