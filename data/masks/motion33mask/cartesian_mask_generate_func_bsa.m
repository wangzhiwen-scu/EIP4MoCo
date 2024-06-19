function [] = cartesian_mask_generate_func_bsa(index, mask_type, subFolderPath)

if mask_type == "random"
      
    z1=length(NH1);
    ranlist1= randperm(z1);  % randperm 随机打乱一个数列函数 y = randperm(n) y是把1到n这些数随机打乱得到的一个数字序列
    z2=length(NH2);
    ranlist2= randperm(z2); 

    o=zeros(M,N);
    o((NH1(end)+1):(NH2(1)-1),:)=1;

    ranrow1=ranlist1(1:Num_Ran/2);
    o(ranrow1,:)=1;
    ranrow2=ranlist2(1:Num_Ran/2)+length(NH1)+Num_fixed;
    o(ranrow2,:)=1;
elseif mask_type == "equidistant"
    %  mask_type, center_ratio

    Num_fixed=15;   %中心区域行数 1%=1+2; 5% = 8+5, 10% = 14+10, 15%=24+12, 20% = 
    Num_Ran=60;     %其他区域随机行数

 
    M=240; 
    N=240;   %采样模板大小(M+2,N+2)

    Mf=round(M/2); % Divides the mask into upper (NH1) and lower (NH2) halves excluding the central region.
    Nf=round(N/2);

    NH1=1:(Mf-Num_fixed/2); % These lines create two sets of indices representing the upper (NH1) and lower (NH2) halves of the k-space mask, excluding the central region.
    NH2=(Mf+Num_fixed/2+1):M;
    
    % Compute equidistant row indices in each half
    equi_rows1 = round(linspace(1, length(NH1), Num_Ran/2));
    equi_rows2 = round(linspace(1, length(NH2), Num_Ran/2));

    o=zeros(M,N);
    o((NH1(end)+1):(NH2(1)-1),:)=1;

    % Sample the equidistant rows in each half
    o(NH1(equi_rows1),:)=1;
    o(NH2(equi_rows2),:)=1;
elseif mask_type == "gaussian"
    %  mask_type, center_ratio

    Num_fixed=15;   %中心区域行数 1%=1+2; 5% = 8+5, 10% = 14+10, 15%=24+12, 20% = 
    Num_Ran=55;     %其他区域随机行数

 
    M=240; 
    N=240;   %采样模板大小(M+2,N+2)

    Mf=round(M/2); % Divides the mask into upper (NH1) and lower (NH2) halves excluding the central region.
    Nf=round(N/2);

    NH1=1:(Mf-Num_fixed/2); % These lines create two sets of indices representing the upper (NH1) and lower (NH2) halves of the k-space mask, excluding the central region.
    NH2=(Mf+Num_fixed/2+1):M;
    
    % Generate a random mean between the limits of NH1 and NH2
    random_mean = rand * (NH2(1) - NH1(end)) + NH1(end);

    % Generate a standard Gaussian distribution for the whole mask with the random mean
    pd = makedist('Normal','mu',random_mean,'sigma',M/4);

    % Generate the cumulative distribution function (CDF)
    cdf_vals = cdf(pd,1:M);

    % Find rows that correspond to the linearly spaced CDF values
    % This will give a row selection according to the Gaussian distribution
    rows = arrayfun(@(x) find(cdf_vals > x, 1, 'first'), linspace(0,1,Num_Ran), 'UniformOutput', false);

    % Convert cell array of indices to numeric array
    rows = cell2mat(rows);

    o=zeros(M,N);
    o((NH1(end)+1):(NH2(1)-1),:)=1;

    % Sample the rows corresponding to the Gaussian distribution
    o(rows,:) = 1;
end
    
Umask=o;

o1=zeros(M+2,N+2);
o1(2:end-1,2:end-1)=o;
whos o1

disp('测量比例，即采样率')
disp(sum(sum(o1))/((M+2)*(N+2)))
  
% figure;imshow(o1)
% title('平行线采样模板')
%  保存
fileName = strcat('cartesian_ablation2_240_240_',string(index));
filePath = fullfile(subFolderPath, fileName);
save(filePath, 'Umask');

end
% I want to you modify this function to generate a mask with a fixed central 
% region and a standard gaussian sampling region outside the central region,
% while original function is random  sampling region outside the central region.

