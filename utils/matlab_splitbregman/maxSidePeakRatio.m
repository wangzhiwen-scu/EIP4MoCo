function [maxSPR,SPR] = maxSidePeakRatio(R,varargin)
% function [maxSPR,SPR] = maxSidePeakRatio(R,opt)
% function [maxSPR,SPR] = maxSidePeakRatio(R,opt,selecPixels)
% function [maxSPR,SPR] = maxSidePeakRatio(R,opt,selecPixels,modePlot)
%
% Computes the maximum sidelobe to peak ratio where SPR=|PSF(i,j)/PSF(i,i)|
% for an undersampled FFT where R provides the undersampling
% 
% R = matrix of zeros and ones that provides the undersampling, which has
% the same dimension as the image

modePlot = 'n';

if (nargin <= 1)
    opt     = '0';
else
    opt     = varargin{1};
end

if (nargin >= 4), modePlot = varargin{3}; end;

uS      = size(R);

switch opt
    case 0 % one point only 
        selecPixels = 1; % selected pixel
        uNum    = 1;
        numAll  = 1:prod(uS); % all pixels
    case 1 % for all pixels
        uNum    = prod(uS);
        numAll  = 1:prod(uS);
        selecPixels = numAll;
    case 2
        selecPixels = varargin{2};
        uNum    = length(selecPixels);
        numAll  = 1:prod(uS);
end % nargin

if (modePlot == 'y')
    figure; spy(R); title('Undersampling');
end % modePlot

SPR     = zeros(uS);

for ie = 1:uNum
    % Each pixel
    u   = zeros(uS);
    u(selecPixels(ie)) = 1;
    % FFT
    F   = R.*fft2(u);
    %figure; imagesc(abs(fftshift(F)));
    % IFFT
    uRec = ifft2(F);
    if (modePlot == 'y')
        figure; surf(abs(uRec));
    end % modePlot
    PSFThis = uRec(setdiff(numAll,selecPixels(ie)));
    SPR(ie) = max(abs(PSFThis(:)/uRec(selecPixels(ie))));
    %     
end % ie
maxSPR = max(SPR(:));
