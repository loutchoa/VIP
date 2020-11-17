function MB = multiscalelaplacianblob(I)
%   MB = multiscalelaplacianblob(I)
%
% Laplacian multi-scale blob detector. Detects both bright and dark blobs



% Convert to gray scale
if ndims(I)==3
    I = rgb2gray(I);
elseif ndims(I) > 3
    error('Can only handle 2D gray scale and 3 channel color images');
end

% Parameters

% Define scale range
NoScales = 31; 
sigma0 = 1.5; 
k = 1.1; % Scale level steps
disp('Min scales: ')
sigmaD = sigma0
disp('Max scales: ')
sigmaD = k^(NoScales-1) * sigma0

% Define threshold
%Hthres = 1500; 
thres = 30; % 20



If = fft2(double(I));

FeatureStrength = zeros( size(I,1), size(I,2), NoScales);

% Build the scale space of scale normalized Laplacian response
for n=1:NoScales
    sigmaD = k^(n-1) * sigma0;

    Lxx = real(ifft2(scale(If, sigmaD, 0, 2)));
    Lyy = real(ifft2(scale(If, sigmaD, 2, 0)));
    FeatureStrength(:, :, n) = sigmaD^2*(Lxx + Lyy);
end
    
    
M = LocalMaxima3DFast(-FeatureStrength);
idx=find(M > 0);
idxidx = find(abs(M(idx)) > thres);
[r, c, s] = ind2sub(size(M), idx(idxidx));
blobmaxima = [r, c, k.^(s-1) * sigma0];

M = LocalMaxima3DFast(FeatureStrength);
idx=find(M > 0);
idxidx = find(abs(M(idx)) > thres);
[r, c, s] = ind2sub(size(M), idx(idxidx));
blobminima = [r, c, k.^(s-1) * sigma0];

MB = [blobmaxima; blobminima];

return;

end

