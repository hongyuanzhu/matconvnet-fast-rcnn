%VL_NNPOOL3D CNN poolinng.
%   Y = VL_NNPOOL3D(X, POOL) applies the pooling operator to all
%   channels of the data X using a square filter of size POOL. X is a
%   SINGLE array of dimension H x W x L x D x N where (H,W,L) are the
%   height, width and length of the map stack, D is the image depth (number
%   of feature channels) and N the number of of images in the stack.
%
%   Y = VL_NNPOOL3D(X, [POOLY, POOLX, POOLL]) uses a rectangular filter of
%   height POOLY, width POOLX and length POOLL
%
%   DZDX = VL_NNPOOL3D(X, POOL, DZDY) computes the derivatives of the
%   block projected onto DZDY. DZDX and DZDY have the same dimensions
%   as X and Y respectively.
%
%   VL_NNCONV3D(..., 'option', value, ...) takes the following options:
%
%   `Stride`:: 1
%     The output stride (downsampling factor). It can be either a
%     scalar for isotropic downsampling or a vector [STRIDEY
%     STRIDEX STRIDEL].
%
%   `Pad`:: 0
%     The amount of input padding. Input images are padded with zeros
%     by this number of pixels on all sides before the convolution is
%     computed. It can also be a vector [TOP BOTTOM LEFT RIGHT LENGTH] to
%     specify a different amount of padding in each direction. The
%     size of the poolin filter has to exceed the padding.
%
%   `Method`:: 'max'
%     Specify method of pooling. It can be either 'max' (retain max value
%     over the pooling region per channel) or 'avg' (compute the average
%     value over the poolling region per channel).
%
%   The pooling window must be not larger than the padded image, i.e.
%
%     1 <= POOLY <= HEIGHT + (PADTOP + PADBOTTOM),
%     1 <= POOLX <= WIDTH + (PADLEFT + PADRIGHT).
%     1 <= POOLL <= LENGTH + (PADLENGTH + PADLENGTH)
%
%   The output a is a SINGLE array of dimension YH x YW x YL x K x N of N
%   images with K challens and size:
%
%     YH = floor((H + (PADTOP+PADBOTTOM) - POOLY)/STRIDEY) + 1,
%     YW = floor((W + (PADLEFT+PADRIGHT) - POOLX)/STRIDEX) + 1,
%     YL = floor((L + (PADLENGTH+PADLENGTH) - POOLL)/STRIDEL) + 1.
%
%   The derivative DZDY has the same dimension of the output Y and
%   the derivative DZDX has the same dimension as the input X.
%
%   ## CUDNN SUPPORT not yet supported
%
%
% Copyright (C) 2016 Tuan-Hung VU.
% All rights reserved.
