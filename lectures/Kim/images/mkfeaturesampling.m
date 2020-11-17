% Script for making illustrations for the features I lecture.

close all;
clear all;

I = imread('Erimitage1.png');
J = imread('Erimitage2.png');


% Regular grid
[X,Y] = meshgrid([1:71:size(I,2)], [1: 47: size(I,1)]);
figure(1)
imshow(I)
hold on;
plot(X,Y, 'r.', 'MarkerSize',15)
hold off;
print('-depsc2','Erimitage1_grid.eps')

figure(2)
imshow(J)
hold on;
plot(X,Y, 'r.', 'MarkerSize',15)
hold off;
print('-depsc2','Erimitage2_grid.eps')


% Random locations
x=randi([1, size(I,2)], 100, 1);
y=randi([1, size(I,1)], 100, 1);

figure(3)
imshow(I)
hold on;
plot(x, y, 'r.', 'MarkerSize',15)
hold off;
print('-depsc2','Erimitage1_random.eps')


x=randi([1, size(I,2)], 100, 1);
y=randi([1, size(I,1)], 100, 1);
figure(4)
imshow(J)
hold on;
plot(x, y, 'r.', 'MarkerSize',15)
hold off;
print('-depsc2','Erimitage2_random.eps')



% Sparse points
% Use Harris multi-scale

Ig = rgb2gray(I);
MH = multiscaleharris(Ig, 1); % with localization


figure(5)
imshow(I)
hold on;
plot(MH(:,2), MH(:,1), 'r.', 'MarkerSize',15)
hold off;
print('-depsc2','Erimitage1_harris.eps')

Ig = rgb2gray(J);
MH = multiscaleharris(Ig, 1); % with localization

figure(6)
imshow(J)
hold on;
plot(MH(:,2), MH(:,1), 'r.', 'MarkerSize',15)
hold off;
print('-depsc2','Erimitage2_harris.eps')
