classdef RoiPooling < dagnn.ElementWise
  properties
    method = 'max'
    poolSize = [7 7]
    spatial_scale = 1/16
    argmax = [];
  end

  methods
    function outputs = forward(self, inputs, params)
      [outputs{1}, self.argmax] = vl_nnroipooling(inputs{1}, self.poolSize, inputs{2}, 'spatialscale', self.spatial_scale);
    end

    function [derInputs, derParams] = backward(self, inputs, params, derOutputs)
      derInputs{1} = vl_nnroipooling(inputs{1}, self.poolSize, inputs{2}, derOutputs{1}, self.argmax, 'spatialscale', self.spatial_scale) ;
      derInputs{2} = [];
      derParams = {} ;
    end

    function obj = RoiPooling(varargin)
      obj.load(varargin) ;
    end
  end
end
