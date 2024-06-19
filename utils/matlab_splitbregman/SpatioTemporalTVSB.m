% [u] = SpatioTemporalTVSB(RAll,fAll,dimIm,betaxy,betat,mu,lambda,
% gamma, nInner, nBreg)
% [u, errAll] = SpatioTemporalTVSB(RAll,fAll,betaxy,betat,dimIm,mu,
% lambda,gamma,nInner,nBreg,imTrueAll)
%
% Inputs:
%
% RAll      = undersampling matrix, same size as fAll
% fAll      = 2D+time data, which corresponds to fft2(imTrueAll)+noise
% dimIm     = Nx*Ny*frames
% betaxy    = parameter weighting the spatial TV term (sparsiy on the
% spatial domain), use=1 and tune depending of the problem
% betat     = parameter weighting the temporal TV term (sparsiy on the
% temporal domain), use=1 and tune depending of the problem
% mu        = parameter weighting the data fidelity term, use mu=1
% lambda    = parameter weighting the TV constraints, use lambda=1
% gamma     = parameter to improve the conditioning, use between mu/100 and
% mu
% nInner    = inner iterations, use n=1
% nBreg     = number of (outer) iterations
% imTrueAll = target image to compute the error at each iteration
% 
% Outputs: 
%
% u         = reconstructed image, size dimIm
% errAll    = Relative solution error norm at each iteration for all frames
%
% Spatiotemporal total variation (ST-TV) using the Split Bregman
% formulation. ST-TV minimizes
%       min_u |grad_x,y u|_1 + |grad_t u|_1 st. ||Fu-f||^2 < delta, 
% (for more details, see the following paper) 
% If you use this code, please, cite the following paper: P Montesinos, 
% JFPJ Abascal, L Cussó, J J Vaquero, M Desco. Application of the
% compressed sensing technique to self-gated cardiac cine sequences in
% small animals. Magn Reson Med., 72(2): 369–380, 2013. DOI:
% http://dx.doi.org/10.1002/mrm.24936    
%
% This code is an extension to the temporal dimension 
% of spatial TV Goldstein'n code mrics.m downloaded from  
% (http://www.ece.rice.edu/~tag7/Tom_Goldstein/Split_Bregman.html), see Tom
% Goldstein and Stanley Osher. The Split Bregman Method for L1-Regularized
% Problems. SIAM J. Imaging Sci., 2(2), 323–343.  
%
%
% Juan Felipe Pérez-Juste Abascal, Paula Montesinos
% Departamento de Bioingeniería e Ingeniería Aeroespacial
% Universidad Carlos III de Madrid, Madrid, Spain
% paumsdv@gmail.com, juanabascal78@gmail.com, desco@hggm.es

function [u,varargout] = SpatioTemporalTVSB(RAll,fAll, dimIm,betaxy,betat,mu, lambda, gamma, nInner, nBreg,varargin)

rows        = dimIm(1);
cols        = dimIm(2);
numTime     = dimIm(3);

% normalize the data so that standard parameter values work
normFactor  = getNormalizationFactor(RAll(:,:,1),fAll(:,:,1));
fAll        = normFactor*fAll;

% Reserve memory for the auxillary variables
f0All       = fAll;
u           = zeros(rows,cols,numTime);
x           = zeros(rows,cols,numTime);
y           = zeros(rows,cols,numTime);
bx          = zeros(rows,cols,numTime);
by          = zeros(rows,cols,numTime);
t           = zeros(rows,cols,numTime);
bt          = zeros(rows,cols,numTime);

uBest       = u;
errBest     = inf;

% RHS of the linear system
scale       = sqrt(rows*cols);
murf        = ifft2(mu*fAll)*scale;

switch nargin
    case 11
        errAll      = zeros(nBreg,numTime);
        imTrueAll   = varargin{1};
        imTrueAll   = (normFactor*scale)*imTrueAll;
        for it = 1:numTime
            imTrueAllNorm(it) = norm(reshape(imTrueAll(:,:,it),[],1));
        end
end % nargin

% Build Kernels 
% Spatiotemporal Hessian in the Fourier Domain
uker = zeros(rows,cols,numTime);
uker(1,1,1) = 6; uker(1,2,1)=-1; uker(2,1,1)=-1;
uker(rows,1,1)=-1; uker(1,cols,1)=-1;
uker(1,1,2)=-1; uker(1,1,numTime)=-1;
uker        = lambda*fftn(uker) + gamma + mu*RAll;

h   = waitbar(0);
h2  = figure;

%  Do the reconstruction
for outer = 1:nBreg;
    for inner = 1:nInner;
        % update u
        % For each time
        rhs     = murf+lambda*Dxt(x-bx)+lambda*Dyt(y-by)+lambda*Dtt(t-bt)+gamma*u;
        
        % Reconstructed image solving the equation in 3D
        u       = ifftn(fftn(rhs)./uker);
        
        % update x and y
        dx      = Dx(u);
        dy      = Dy(u);
        dt      = Dt(u);
        [x,y]   = shrink2(dx+bx,dy+by,betaxy/lambda);        
        t       = shrink1(dt+bt,betat/lambda); 
        
        % update bregman parameters
        bx      = bx+dx-x;
        by      = by+dy-y;
        bt      = bt+dt-t;
    end % inner
    
    fForw       = RAll.*fft2(u)/scale;
    fAll        = fAll + f0All - fForw;    
    murf        = ifft2(mu*fAll)*scale;
    
    if (nargin >= 11)
        % Compute the error
        for it = 1:numTime
            errThis = norm(reshape(imTrueAll(:,:,it)-u(:,:,it),[],1))./imTrueAllNorm(it);
            errAll(outer,it) = errThis;
        end
        
        if (mean(errThis) <= errBest)
            uBest       = u;
            errBest     = mean(errThis);
        end %errThis
        
        if any([outer==1 rem(outer,20)==0])
            figure(h); waitbar(outer/nBreg,h);    
            figure(h2);
            imagesc(abs(u(:,:,1))); title(['ST-TV iter. ' num2str(outer) ]); colormap gray; axis image; drawnow;
        end        
    end % nargin
    
end % outer

close(h);

if (nargin >= 9)
    u = uBest;
    varargout{1} = errAll;
end % nargin

% undo the normalization so that results are scaled properly
u = u/(normFactor*scale);

return;


function normFactor = getNormalizationFactor(R,f)

normFactor = 1/norm(f(:)/size(R==1,1));

return;


function d = Dx(u)
[rows,cols,time] = size(u);
d = zeros(rows,cols,time);
d(:,2:cols,:) = u(:,2:cols,:)-u(:,1:cols-1,:);
d(:,1,:) = u(:,1,:)-u(:,cols,:);
return

function d = Dxt(u)
[rows,cols,time] = size(u);
d = zeros(rows,cols,time);
d(:,1:cols-1,:) = u(:,1:cols-1,:)-u(:,2:cols,:);
d(:,cols,:) = u(:,cols,:)-u(:,1,:);
return

function d = Dy(u)
[rows,cols,time] = size(u);
d = zeros(rows,cols,time);
d(2:rows,:,:) = u(2:rows,:,:)-u(1:rows-1,:,:);
d(1,:,:) = u(1,:,:)-u(rows,:,:);
return

function d = Dyt(u)
[rows,cols,time] = size(u);
d = zeros(rows,cols,time);
d(1:rows-1,:,:) = u(1:rows-1,:,:)-u(2:rows,:,:);
d(rows,:,:) = u(rows,:,:)-u(1,:,:);
return

function d = Dt(u) % Time derivative for 3D matrix
[rows,cols,time] = size(u);
d = zeros(rows,cols,time);
d(:,:,2:time) = u(:,:,2:time)-u(:,:,1:time-1);
d(:,:,1) = u(:,:,1)-u(:,:,time);
return

function d = Dtt(u) % Time derivative for 3D matrix, transpose
[rows,cols,time] = size(u);
d = zeros(rows,cols,time);
d(:,:,1:time-1) = u(:,:,1:time-1)-u(:,:,2:time);
d(:,:,time) = u(:,:,time)-u(:,:,1);
return

function [xs,ys] = shrink2(x,y,lambda)

s = sqrt(x.*conj(x)+y.*conj(y));
ss = s-lambda;
ss = ss.*(ss>0);

s = s+(s<lambda);
ss = ss./s;

xs = ss.*x;
ys = ss.*y;

return;

function xs = shrink1(x,lambda)

s = abs(x);
xs = sign(x).*max(s-lambda,0);







