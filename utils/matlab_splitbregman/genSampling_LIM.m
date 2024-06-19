function [minIntrVec,stat,actpctg] = genSampling_LIM(pdf,iter,tol)
% Based on genSampling.m (SparseMRI V0.2) by Michael Lustig 2007
% http://web.stanford.edu/~mlustig/SparseMRI.html, see M. Lustig, D.L
% Donoho and J.M Pauly "Sparse MRI: The Application of Compressed Sensing
% for Rapid MR Imaging" Magnetic Resonance in Medicine, 2007 Dec;
% 58(6):1182-1195. 
%
%[mask,stat,N] = genSampling_LIM(pdf,iter,tol) 
%
% a monte-carlo algorithm to generate a sampling pattern with
% minimum peak interference. The number of samples will be
% sum(pdf) +- tol
%
%	pdf - probability density function to choose samples from
%	iter - number of tries
%	tol  - the deviation from the desired number of samples in samples
%
% returns:
%	mask - sampling pattern
%	stat - vector of min interferences measured each try
%	actpctg    - actual undersampling factor

pdf(find(pdf>1)) = 1;
K = sum(pdf(:));

minIntr = 1e99;
minIntr_b = 1e99;
%minIntrVec = zeros(size(pdf));
minIntrVec = zeros(length(pdf),length(pdf));
minIntrVec_b = zeros(length(pdf),length(pdf));


for n=1:iter
    tmp_aux = zeros(size(pdf));
    while abs(sum(tmp_aux(:)) - K) > tol
        tmp_aux = rand(size(pdf))<pdf;
    end
    tmp = zeros(length(pdf),length(pdf));
    pdf_aux = zeros(length(pdf),length(pdf));
    for p=1:length(pdf)
        tmp(:,p) = tmp_aux(:);
        pdf_aux(:,p)=pdf(:);
    end
    
    % ----
    %TMP = ifft2(tmp./pdf_aux);
    % 	if max(abs(TMP(2:end))) < minIntr
    % 		minIntr = max(abs(TMP(2:end)));
    %         minIntrVec = tmp;
    %     end    
    %   name=sprintf('%s%s%d%s','sampling','_',n,'.mat');
    %   save(name,'tmp');
    % ----
    [a1,b]=maxSidePeakRatio(fftshift(tmp./pdf_aux), 2,[1 round((length(pdf)^2/2)-length(pdf)/2)],'n');    
    %----
    if a1 < minIntr_b
        minIntr_b = a1;
        minIntrVec = fftshift(tmp);
    end
    
    stat(n) = a1;%max(abs(TMP(2:end)));
    %  stat_b(n) = a1;
end

actpctg = sum(minIntrVec(:))/prod(size(minIntrVec));



