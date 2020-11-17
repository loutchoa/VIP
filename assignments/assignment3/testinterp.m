function testinterp(noiselevel, pointfrac)
% testprogram for interp(.), a TPS-approximation of sparse measurements
    
% This program generates a 4-partite ground truth image of 4
% different values. Normal distributed noise with a standard
% deviation of 'noiselevel' is added.  From this image a fraction
% of pixels (determined by 'pointfrac') is extracted as sparse
% data. Then interp(.) is called with a constant initial value
% determined by 'startval', with 'maxitt' = 400 iterations and with a
% value of 'lambda' = 0.001.
    
% Output is 4 images showing the ground truth and the
% reconstruction in the first row.  The second row shows the
% position of the feature points and the reconstruction error.
  
% Please feel free to use and modify the program, but don't
% complain.  Please note that for other data you may have to tune
% the parameters to get optimal results. 
    
% Definitions
    dimx     = 100;
    dimy     = 100;
    % dens     = 0.1;
    dx2      = dimx/2;
    dy2      = dimy/2;
    startval = 3.0;
    lambda   = 0.01;
    maxitt   = 500;
    
% define ground truth
    GT = zeros(dimx, dimy);
    GT(1:dx2,1:dy2)          = - 10.0;
    GT(1:dx2,dy2+1:end)      =    0.0;
    GT(dx2+1:end,1:dy2)      =    5.0;
    GT(dx2+1:end,dy2+1:end)  =   15.0;

% generate data
    Data     = GT + noiselevel*randn(dimx,dimy);
    points   = (rand(dimx,dimy) < pointfrac);
    values   = (1*points).*Data;
    start    = startval*ones(dimx,dimy);
 
% apply interpolatin
Ipim = interp(start, points, values, maxitt, lambda);

% compare and display result
err = abs(Data - Ipim);
rms = sqrt(sum(sum((Data - Ipim)^2))/(dimx*dimy));
fprintf('average reconstruction error: %8.5f \n', rms);

shGT = GT - min(min(GT));
shGT = shGT/max(max(shGT));
shIp = Ipim - min(min(Ipim));
shIp = shIp/max(max(shIp));
sher = err - min(min(err));
sher = sher/max(max(sher));

h = figure(1);
subplot(2,2,1);
imshow(shGT);
title('Ground truth');
subplot(2,2,2);
imshow(shIp);
title('Reconstruction');
subplot(2,2,3);
imshow(points);
title('Sparse point position');
subplot(2,2,4);
imshow(sher);
title('Reconstruction error');
end


