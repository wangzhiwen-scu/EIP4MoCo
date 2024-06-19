function [u_tv] = mysplitbregman(img, mask)

sim_2dmri = double(img);

data0   = fft2(sim_2dmri);
image0 = sim_2dmri;

mask = double(mask);
mask = fftshift(mask);
RAll = mask;



N           = size(image0);     


data        = data0.*RAll;
% ------------------------------------------------------------------------
% IFFT
fr          = 1;
u_ifft      = ifft2(data);
% ------------------------------------------------------------------------
% STATIC Spatial Total Variation reconstruction using Split Bregman
% Code download from
% http://www.ece.rice.edu/~tag7/Tom_Goldstein/Split_Bregman.html
mu          = 1;
lambda      = 1;
gamma       = 1e-4;
nInner      = 1;
nBreg       = 200;
% imagesc((fftshift(log(abs(data0(:,:,5))))))
% Goldstein's spatial TV using the Split Bregman formulation
% u_tv = mrics(RAll(:,:,1),data(:,:,1), mu, lambda, gamma, nInner, nBreg);
% 
% SpatialTVSB.m: same as mrics.m but it computes the solution error
% norm
% Reconstruction of one slice only
[u_tv,err_tv] = SpatialTVSB(RAll,data, mu, lambda, gamma, nInner, nBreg,image0);

% ------------------------------------------------------------------------
% SpatioTemporal Total Variation with larger temporal sparsity
% Dynamic reconstruction
% betaxy      = 0.3;
% betat       = 0.7;
% mu          = 1;
% lambda      = 1;
% gamma       = 1;
% nInner      = 1;
% nBreg       = 100;
% [u_ttv,err_ttv] = SpatioTemporalTVSB(RAll,data,N,betaxy,betat,mu,lambda,gamma,nInner,nBreg,image0);
% ------------------------------------------------------------------------
% Comparison of results

% figure; 
% tmp     = (image0);
% subplot(2,2,1); imagesc(abs(tmp(40:150,70:190))); axis image; 
% axis off; colormap gray; title('Full data'); ca = caxis;
% tmp     = (u_ifft);
% subplot(2,2,2); imagesc(abs(tmp(40:150,70:190))); axis image; 
% axis off; colormap gray; title('IFFT2'); caxis(ca);
% tmp     = (u_tv);
% subplot(2,2,3); imagesc(abs(tmp(40:150,70:190))); axis image; 
% axis off; colormap gray; title(['STV , ' num2str(100*sparsity) '% undersampling' ]);
% tmp     = (u_tv);
% subplot(2,2,4); imagesc(abs(tmp(40:150,70:190))); axis image; 
% axis off; colormap gray; title(['STTV, ' num2str(100*sparsity) ' % undersampling' ]);
% caxis(ca);
% drawnow; 

% Minimum error norm
% norm(u_tv(:)-reshape(image0,[],1))/norm(reshape(image0,[],1))
% norm(reshape(u_ttv,[],1)-reshape(image0,[],1))/norm(reshape(image0,[],1))

% Convergence
% figure, plot(err_tv); hold on; plot(err_tv,'--r'); 
% xlabel('Iteration number'); ylabel('Relative solution error norm'); 
% legend('S-TV','ST-TV');

% save('my_mat.mat', 'u_tv')

%
end

