% Demo_SpatioTemporalTV_SplitBregman.m
%
% If you use this code, please, cite the following paper: P Montesinos, 
% JFPJ Abascal, L Cussó, J J Vaquero, M Desco. Application of the
% compressed sensing technique to self-gated cardiac cine sequences in
% small animals. Magn Reson Med., 72(2): 369–380, 2013. DOI:
% http://dx.doi.org/10.1002/mrm.24936 
%
% Code downloaded from repository: 
% https://github.com/HGGM-LIM/Split-Bregman-ST-Total-Variation-MRI
% -------------------------------------------------------------------------
% Demo for reconstructing cardiac cine MRI data with spatial total
% variation (S-TV) and spatiotemporal total variation (ST-TV) using the
% Split Bregman formulation. 
% 
% S-TV is solved using Goldstein'n code mrics.m downloaded from 
% (http://www.ece.rice.edu/~tag7/Tom_Goldstein/Split_Bregman.html), see Tom
% Goldstein and Stanley Osher. The Split Bregman Method for L1-Regularized
% Problems. SIAM J. Imaging Sci., 2(2), 323–343.  
%
% ST-TV is solved using a modified version of S-TV that minimizes
% min_u betaxy|grad_x,y u|_1 + betat|grad_t u|_1 st. ||Fu-f||^2 < delta, proposed in 
% P Montesinos et al. Magn Reson Med., 72(2): 369–380, 2013.   
%
% Data corresponds to a prospective cardiac cine MRI study on a healthy rat 
% (one slice and 16 frames). 
%
% Undersampling is simulated using a modified version of Lustig's variable
% density pdf, downloaded from (SparseMRI V0.2)
% http://web.stanford.edu/~mlustig/SparseMRI.html, see M. Lustig, D.L
% Donoho and J.M Pauly "Sparse MRI: The Application of Compressed Sensing
% for Rapid MR Imaging" Magnetic Resonance in Medicine, 2007 Dec;
% 58(6):1182-1195.   
%
% Juan FPJ Abascal, Paula Montesinos
% Departamento de Bioingeniería e Ingeniería Aeroespacial
% Universidad Carlos III de Madrid, Madrid, Spain
% paumsdv@gmail.com, juanabascal78@gmail.com, desco@hggm.es

% 
% Load data: Simulated absolute image from the acquired retrospective
% cardiac cine data set with 8 frames.  
% The acquired data set is available from
% http://biig.uc3m.es/cardiac-cine-data/
load('dataCine8fr','image0'); 

% Simulate data
data0   = fft2(image0);         

% Display images
figure; 
count       = 0;
for it = 1:size(image0,3)
    count       = count+1;
    subplot(3,3,count); imagesc(abs((image0(:,:,it)))); 
    axis image; axis off; colorbar; title(['image fr ' num2str(it)]); colormap gray;
end

N           = size(image0);     
% ------------------------------------------------------------------------
% Simulate data and undersampling pattern
%
% Parameters for Lustig's variable density pdf
rand('state',1);
DN          = [N(1),1];     
%   sparsity    = 0.2; radius = 0; P = 4;            % x5
    sparsity    = 0.15; radius = 0; P = 6;           % x7 
%   sparsity    = 0.1; radius = 0; P = 9;           % x10 
pdf         = genPDF(DN,P,sparsity,2,radius,0);    % generates the sampling PDF
for it = 1:size(image0,3)
    temp    = genSampling_LIM(pdf,10,1);           % generates sampling pattern
    indR    = temp(:,1)==1;
    R       = zeros(N(1:2));
    R(indR,:) = 1;
%   figure; spy(R), 100*nnz(R(:,1))/N(1)        
    RAll(:,:,it)        = R;
end % it

data        = data0.*RAll;
% ------------------------------------------------------------------------
% IFFT
fr          = 1;
u_ifft      = ifft2(data(:,:,fr));
% ------------------------------------------------------------------------
% STATIC Spatial Total Variation reconstruction using Split Bregman
% Code download from
% http://www.ece.rice.edu/~tag7/Tom_Goldstein/Split_Bregman.html
mu          = 1;
lambda      = 1;
gamma       = 1e-4;
nInner      = 1;
nBreg       = 100;

% Goldstein's spatial TV using the Split Bregman formulation
% u_tv = mrics(RAll(:,:,1),data(:,:,1), mu, lambda, gamma, nInner, nBreg);
% 
% SpatialTVSB.m: same as mrics.m but it computes the solution error
% norm
% Reconstruction of one slice only
[u_tv,err_tv] = SpatialTVSB(RAll(:,:,fr),data(:,:,fr), mu, lambda, gamma, nInner, nBreg,image0(:,:,fr));
% ------------------------------------------------------------------------
% SpatioTemporal Total Variation with larger temporal sparsity
% Dynamic reconstruction
betaxy      = 0.3;
betat       = 0.7;
mu          = 1;
lambda      = 1;
gamma       = 1;
nInner      = 1;
nBreg       = 100;
[u_ttv,err_ttv] = SpatioTemporalTVSB(RAll,data,N,betaxy,betat,mu,lambda,gamma,nInner,nBreg,image0);
% ------------------------------------------------------------------------
% Comparison of results
figure; 
tmp     = (image0(:,:,fr));
subplot(2,2,1); imagesc(abs(tmp(40:150,70:190))); axis image; 
axis off; colormap gray; title('Full data'); ca = caxis;
tmp     = (u_ifft);
subplot(2,2,2); imagesc(abs(tmp(40:150,70:190))); axis image; 
axis off; colormap gray; title('IFFT2'); caxis(ca);
tmp     = (u_tv);
subplot(2,2,3); imagesc(abs(tmp(40:150,70:190))); axis image; 
axis off; colormap gray; title(['STV , ' num2str(100*sparsity) '% undersampling' ]);
tmp     = (u_ttv(:,:,fr));
subplot(2,2,4); imagesc(abs(tmp(40:150,70:190))); axis image; 
axis off; colormap gray; title(['STTV, ' num2str(100*sparsity) ' % undersampling' ]);
caxis(ca);
drawnow; 

% Minimum error norm
norm(u_tv(:)-reshape(image0(:,:,fr),[],1))/norm(reshape(image0(:,:,fr),[],1))
norm(reshape(u_ttv(:,:,fr),[],1)-reshape(image0(:,:,fr),[],1))/norm(reshape(image0(:,:,fr),[],1))

% Convergence
figure, plot(err_tv); hold on; plot(err_ttv(:,fr),'--r'); 
xlabel('Iteration number'); ylabel('Relative solution error norm'); 
legend('S-TV','ST-TV');


%