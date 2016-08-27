%VL_NNCONV3D CNN convolution.
%   Y = VL_NNCONV3D(X, F, B) computes the 3D convolution of the image 
%   stack X with the filter bank F and biases B. If B is the empty matrix,
%   then no biases are added. If F is the empty matrix, then the function 
%   does not filter the image, but still adds the biases as well as 
%   performing downsampling and padding as explained below.
%
%   X is a SINGLE array of dimension H x W x L x D x N where (H,W) are
%   the height and width of the image stack, L is the length of stack, 
%   D is the image depth (number of feature channels) and N the number 
%   of stacks.
%
%   F is a SINGLE array of dimension FW x FH x FL x FD x K where (FH,FW,FL)
%   are the filter height and width and length and K the number o filters 
%   in the bank. FD is the depth of each filter and must match the depth D of
%   X. Alternatively, FD can *divide* the depth D; in this case,
%   filters are assumed to form G=D/FD *groups* of equal size (where
%   G must divide K). Each group of filters works on a consecutive
%   subset of feature channels of the input array X.
%
%   [DZDX, DZDF, DZDB] = VL_NNCONV3D(X, F, B, DZDY) computes the
%   derivatives of the block projected onto DZDY. DZDX, DZDF, and
%   DZDB, and DZDY have the same dimensions as X, F, B, and Y
%   repsectively. In particular, if B is the empty matrix, then DZDB
%   is also empty.
%
%   VL_NNCONV(..., 'option', value, ...) takes the following options:
%
%   `Stride`:: 1
%     The output stride or downsampling factor. If the value is a
%     scalar, then the same stride is applied to both vertical, horizontal 
%     and temporal directions; otherwise, passing [STRIDEY STRIDEX STRIDEL]
%     allows specifying different downsampling factors for each
%     direction.
%
%   `Pad`:: 0
%     The amount of input padding. Input images are padded with zeros
%     by this number of pixels before the convolution is
%     computed. Passing [TOP BOTTOM LEFT RIGHT TEMPORAL] allows specifying
%     different padding amounts for the top, bottom, left, right and temporal
%     sides respectively. Passing a single scalar applies the same
%     padding to all borders.
%
%   The filter size must be not larger than the padded image, i.e.
%
%     1 <= FH <= H + 2*(PADTOP+PADBOTTOM),
%     1 <= FW <= W + 2*(PADLEFT+PADRIGHT),
%     1 <= FL <= L + 2*(PADTEMPORAL+PADTEMPORAL).
%
%   The output a is a SINGLE array of dimension YH x YW x YL x K x N of
%   N images with K challens and size:
%
%     YH = floor((H + (PADTOP+PADBOTTOM) - FH)/STRIDEY) + 1,
%     YW = floor((W + (PADLEFT+PADRIGHT) - FW)/STRIDEX) + 1.
%     YL = floor((L + (PADTEMPORAL+PADTEMPORAL) - FL)/STRIDEL) + 1.
%
%   ## CUDNN SUPPORT: supported
%

% Copyright (C) 2016 Tuan-Hung VU.
% All rights reserved.
