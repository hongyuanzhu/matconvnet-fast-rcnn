function Y = vl_nnsmoothl1loss(X,c,w,dzdy)
% vl_nnsmoothl1loss - smooth L1 loss
%    Y = VL_NNSMOOTHL1LOSS(X, C) applies the smooth L1 loss on the data X.
%    X has dimension 1 x 1 x D x N, packing N arrays of D-dimensional
%    vectors.
%
%    C contains the regression targets, which should be single. C has
%    dimension 1 x 1 x D x N, corresponding to N arrays of D-dimensional
%    regression targets. In the case of fast-rcnn, regression target is a 
%    4-dimensional vector [dx dy dw dh] corresponding to translation and 
%    scale parameters. D equals 4 x number of object classes.
%
%    W are binary vectors specifying (at most 4) active targets per roi.
%
%    DZDX = VL_NNSMOOTHL1LOSS(X, C, DZDY) computes the derivative DZDX
%    of the CNN with respect to the input X given the derivative DZDY
%    with respect to the block output Y. DZDX has the same dimension
%    as X.
%
% --------------------------------------------
% Reimplementation based on Python Fast R-CNN 
% (https://github.com/rbgirshick/fast-rcnn) 
% Copyright (C) 2016 Tuan-Hung VU.
% All rights reserved.
% --------------------------------------------

sz = [size(X,1) size(X,2) size(X,3) size(X,4)] ;

X_ = gather(X);

c_ = reshape(c, sz);
w_ = reshape(w, sz);

out = zeros(size(X_));

diff = X_ - c_;
diff = w_.*diff;
abs_diff = abs(diff);

n = sz(1)*sz(2) ;
if nargin <= 3
  out = abs_diff - 0.5;
  M = abs_diff < 1;
  out(M) = 0.5*diff(M).^2;
  Y = sum(out(:))/n;
else
  out = diff;
  M = abs_diff >= 1;
  out(M) = (diff(M) > 0) - (diff(M) < 0);
  Y = out * (dzdy / n);
  if isa(X, 'gpuArray')
      Y = gpuArray(Y);
  end
end
