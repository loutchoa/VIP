function maximaSelection = ImpHarris(I, sigmaD, sigmaI);
% Improved Harris Detector

fftI = fft2(I);

Ix = real(ifft2(scale(fftI,sigmaD,1,0)));
Iy = real(ifft2(scale(fftI,sigmaD,0,1)));

Ax =  real(ifft2(scale(fft2(Ix.^2 ),sigmaI,0,0)));
Ay =  real(ifft2(scale(fft2(Iy.^2 ),sigmaI,0,0)));
Axy = real(ifft2(scale(fft2(Ix.*Iy),sigmaI,0,0)));

FeatureStrength = Ax.*Ay - Axy.^2 - 0.06* (Ax + Ay).^2;

%maxima = findLocalMin(-FeatureStrength,8);
M = LocalMaxima2DFast(FeatureStrength);
idx=find(M > 0);
[r, c] = ind2sub(size(M), idx);
maxima = [r, c];


for(i=1:length(maxima(:,1)));
    maximaFeatureStrength(i,1)=FeatureStrength(maxima(i,1),maxima(i,2));
end

%threshold = 0.01*max(maximaFeatureStrength);
%threshold = 0;
threshold = 1000;

selection = (sign(maximaFeatureStrength-threshold)+1)/2;

maximaSelection = [nonzeros(selection.*maxima(:,1)) nonzeros(selection.*maxima(:,2))];