% Script for making illustrations for the features I lecture.

close all;
clear all;

I = imread('Erimitage1.png');
J = imread('Erimitage2.png');

% Gray scale Erimitage1
figure(1)
imshow(rgb2gray(I))
print('-depsc2', 'Erimitage1_gray.eps');


% Image as mesh plot
figure2 = figure(2)
[R, C] = meshgrid(1:size(I,2), 1:size(I,1));
colormap('gray');
axes1 = axes('Parent',figure2);
view(axes1,[-137.5 54]);
hold(axes1,'all');
mesh(R, C, rgb2gray(I), double(rgb2gray(I)),'Parent',axes1);
grid off;
print('-depsc2', 'Erimitage1_gray_mesh.eps');




% Corner cut-out
figure(3)
imshow(I(109:140, 285:330,:))
print('-depsc2', 'Erimitage_corner_template.eps')

figure(4)
imshow(J(93:(93+32-1), 278:(278+46-1),:))
print('-depsc2', 'Erimitage2_corner_template.eps')


% Scale space
% If = fft2(double(rgb2gray(I)));
% L1 = real(ifft2(scale(If, 4, 0,0)));
% L2 = real(ifft2(scale(If, 8, 0,0)));
% L3 = real(ifft2(scale(If, 16, 0,0)));
% 
% figure(4)
% warp(zeros(size(I,1), size(I,2)), rgb2gray(I))
% hold on
% warp(4*ones(size(I,1), size(I,2)), L1)
% warp(8*ones(size(I,1), size(I,2)), L2)
% warp(16*ones(size(I,1), size(I,2)), L3)
% hold off;


% Improved Harris corner (fixed scale)
sigmaI = 1.5;
sigmaD = 0.7 * sigmaI;
maximaSelection = ImpHarris(rgb2gray(I), sigmaD, sigmaI);

figure(5), 
clf;
imshow(I)
hold on;
plot(maximaSelection(:,2), maximaSelection(:,1), 'r.', 'MarkerSize',10)
hold off;
print('-depsc2','Erimitage1_ImpHarris.eps')

maximaSelection = ImpHarris(rgb2gray(J), sigmaD, sigmaI);

figure(6), 
clf;
imshow(J)
hold on;
plot(maximaSelection(:,2), maximaSelection(:,1), 'r.', 'MarkerSize',10)
hold off;
print('-depsc2','Erimitage2_ImpHarris.eps')



% Fixed scale blob detection
sigmaD = 1.5;
thres = 20; % 3
If = fft2(double(rgb2gray(I)));
Lxx = sigmaD^2*real(ifft2(scale(If, sigmaD, 0,2)));
Lyy = sigmaD^2*real(ifft2(scale(If, sigmaD, 2,0)));
FeatureStrength = Lxx + Lyy;

M = LocalMaxima2DFast(-FeatureStrength);
idx=find(M > 0);
idxidx = find(abs(M(idx)) > thres);
[r, c] = ind2sub(size(M), idx(idxidx));
blobmaxima = [r, c];

M = LocalMaxima2DFast(FeatureStrength);
idx=find(M > 0);
idxidx = find(abs(M(idx)) > thres);
[r, c] = ind2sub(size(M), idx(idxidx));
blobminima = [r, c];

figure(7), clf;
imshow(rgb2gray(I))
hold on;
plot(blobmaxima(:,2), blobmaxima(:,1), 'r.', 'MarkerSize',10)
plot(blobminima(:,2), blobminima(:,1), 'y.', 'MarkerSize',10)
hold off;
print('-depsc2','Erimitage1_fixedblobs.eps')


If = fft2(double(rgb2gray(J)));
Lxx = sigmaD^2*real(ifft2(scale(If, sigmaD, 0,2)));
Lyy = sigmaD^2*real(ifft2(scale(If, sigmaD, 2,0)));
FeatureStrength = Lxx + Lyy;

M = LocalMaxima2DFast(-FeatureStrength);
idx=find(M > 0);
idxidx = find(abs(M(idx)) > thres);
[r, c] = ind2sub(size(M), idx(idxidx));
blobmaxima = [r, c];

M = LocalMaxima2DFast(FeatureStrength);
idx=find(M > 0);
idxidx = find(abs(M(idx)) > thres);
[r, c] = ind2sub(size(M), idx(idxidx));
blobminima = [r, c];

figure(8), clf;
imshow(rgb2gray(J))
hold on;
plot(blobmaxima(:,2), blobmaxima(:,1), 'r.', 'MarkerSize',10)
plot(blobminima(:,2), blobminima(:,1), 'y.', 'MarkerSize',10)
hold off;
print('-depsc2','Erimitage2_fixedblobs.eps')


% Multi-scale Laplacian blob detection
MB = multiscalelaplacianblob(I);
figure(9), clf()
imshow(rgb2gray(I))
hold on;
plot(MB(:,2), MB(:,1), 'r.', 'MarkerSize',10)
plotcircle([MB(:,2), MB(:,1)], MB(:,3), 'r');
hold off;
print('-depsc2','Erimitage1_multiscale_laplacian_blobs.eps')

MB = multiscalelaplacianblob(J);
figure(10), clf()
imshow(rgb2gray(J))
hold on;
plot(MB(:,2), MB(:,1), 'r.', 'MarkerSize',10)
plotcircle([MB(:,2), MB(:,1)], MB(:,3), 'r');
hold off;
print('-depsc2','Erimitage2_multiscale_laplacian_blobs.eps')

% Multi-scale Harris corner detection
MH = multiscaleharris(rgb2gray(I), 0);
figure(11), clf()
imshow(I)
hold on;
plot(MH(:,2), MH(:,1), 'r.', 'MarkerSize',10)
plotcircle([MH(:,2), MH(:,1)], MH(:,3), 'r');
hold off;
print('-depsc2','Erimitage1_multiscale_harris.eps')

MH = multiscaleharris(rgb2gray(J), 0);
figure(12), clf()
imshow(J)
hold on;
plot(MH(:,2), MH(:,1), 'r.', 'MarkerSize',10)
plotcircle([MH(:,2), MH(:,1)], MH(:,3), 'r');
hold off;
print('-depsc2','Erimitage2_multiscale_harris.eps')



% Make a fake SIFT orientation histogram
x=[ones(1,12), 2, 4, 2, 10, 5, 4, 9, 7, ones(1,12)];
figure(13), clf()
bar(linspace(0, 360, 32), x)
print('-depsc2','SIFT_orientation_histogram.eps')


Pf = fft2(rgb2gray(I(109:140, 285:330,:)));
Px = real(ifft2(scale(Pf, sigmaD, 0,1)));
Py = real(ifft2(scale(Pf, sigmaD, 1,0)));
[X, Y]=meshgrid(0:(size(Pf, 2)-1), 0:(size(Pf, 1)-1));

figure(13), clf()
imagesc(rgb2gray(I(109:140, 285:330,:))), colormap(gray)
hold on;
quiver(X, Y, Px, Py, 1.5, 'r')
hold off;
print('-depsc2','small_patch_gradients.eps')


figure(14), clf()
imagesc(I(109:140, 285:330,:))
print('-depsc2','small_patch_rgb.eps')

