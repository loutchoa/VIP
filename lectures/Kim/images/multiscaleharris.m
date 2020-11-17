function MH = multiscaleharris (I, localize)
% MH = multiscaleharris (I, localize)
%
%   I        - Gray scale image (2D matrix - can handle uint8 matrices)
%   localize - If 1 then apply localization algorithm
%   MH       - Harris multiscale corner points [row, column, scale]
%
%   Detects multiscale Harris corners.
%
%  Copyright (C) 2011 Kim Steenstrup Pedersen (kimstp@diku.dk)
%  Released under the terms of the GNU Lesser General Public License as 
%  published by the Free Software Foundation. See the source file for full
%  details.

%  lindebergcorner version 1.0                                            
%  Copyright (C) 2011 Kim Steenstrup Pedersen (kimstp@diku.dk)              
%  Department of Computer Science, University of Copenhagen, Denmark        
%                                                                            
%  This file is part of the multi-scale Harris detector package version 1.0 
%                                                                             
%  The multi-scale Harris detector package is free software: you can        
%  redistribute it and/or modify it under the terms of the GNU Lesser       
%  General Public License as published by the Free Software Foundation,      
%  either version 3 of the License, or (at your option) any later version.                                      *
%                                                                         
%  The multi-scale Harris detector package is distributed in the hope that  
%  it will be useful, but WITHOUT ANY WARRANTY; without even the implied    
%  warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the 
%  GNU Lesser General Public License for more details.                      
%                                                                             
%  You should have received a copy of the GNU Lesser General Public License 
%  along with the multi-scale Harris detector package. If not, see          
%  <http://www.gnu.org/licenses/>.           
% 


% Fourier transform the image
If = fft2(I);


% Define scale range
NoScales = 31; 
sigma0 = 1.5; 
alpha = 0.06; 
k = 1.1; % Scale level steps
disp('Min scales: ')
sigmaI = sigma0
sigmaD = 0.7 * sigmaI
disp('Max scales: ')
sigmaI = k^(NoScales-1) * sigma0
sigmaD = 0.7 * sigmaI

% Define threshold
Hthres = 1500; 

H = zeros( size(I,1), size(I,2), NoScales);
Lx = zeros( size(I,1), size(I,2), NoScales);
Ly = zeros( size(I,1), size(I,2), NoScales);
for n=1:NoScales
    sigmaI = k^(n-1) * sigma0;
    sigmaD = 0.7 * sigmaI;

    Lx(:, :, n) = real(ifft2(sigmaD.*scale(If, sigmaD, 0, 1)));
    Ly(:, :, n) = real(ifft2(sigmaD.*scale(If, sigmaD, 1, 0)));

    % Compute the second order structure matrix (structure tensor)
    % T = [A B;
    %      B C]
    A = real(ifft2(scale(fft2(Lx(:, :, n).^2), sigmaI, 0,0)));
    B = real(ifft2(scale(fft2(Lx(:, :, n).*Ly(:, :, n)), sigmaI, 0,0)));
    C = real(ifft2(scale(fft2(Ly(:, :, n).^2), sigmaI, 0,0)));
    
    
    % Harris feature strength - scale normalized but non-gamma normalized
    H(:, :, n) = A .* C - B.^2 - alpha * (A + C).^2;
        
end


% Harris scale selection
M = LocalMaxima3DFast(H); % 26 neighbor search
idx=find(M > Hthres);
[r, c, s] = ind2sub(size(M),idx);
% Remove points that are at the boundary, i.e. 2*sigmaI from
% the boundary
idx = find((r > round(2 * k.^(s-1) * sigma0)) & (r < (size(I,1)-round(2 * k.^(s-1) * sigma0))) & (c > round(2 * k.^(s-1) * sigma0)) & (c < (size(I,2)-round(2 * k.^(s-1) * sigma0))));
MH = [r(idx), c(idx), s(idx)];




% Implement Lindeberg localization scheme for multiscale Harris corners
if localize == 1
    disp('Applying localization algorithm');
   
    % Lindeberg corners
    MHtmp = zeros(size(MH));
    counter = 0;
    for n = 1:size(MH,1) % For each candidate point
        sigmaL = 0.7 * k^(MH(n,3)-1) * sigma0; % Integration window scale (fixed) 
        
        % Make Gaussian window
        [gx,gy] = meshgrid(-round(2*sigmaL):1:round(2*sigmaL), -round(2*sigmaL):1:round(2*sigmaL));
        W = exp(-(gx.^2 + gy.^2)./(2*sigmaL^2));
        W = W ./ sum(W(:));

        counter = counter + 1;
        MHtmp(counter,:) = MH(n,:);

        iter = 1;        
        while iter > 0
        
            % Compute scale space signature
            dmin = zeros(MH(n,3),1);
            A = zeros(2, 2, MH(n,3));
            b = zeros(2, MH(n,3));
 
            % Calculate ROI indices
            ROIRows = (round(MHtmp(counter,1)) - round(2*sigmaL)):1:(round(MHtmp(counter,1)) + round(2*sigmaL));
            ROICols = (round(MHtmp(counter,2)) - round(2*sigmaL)):1:(round(MHtmp(counter,2)) + round(2*sigmaL));

            [X,Y] = meshgrid(ROICols, ROIRows); 

            for s = 1 : MH(n,3) % For all scales below and including this points detection scale             
                                
                A1 = sum(sum(Lx(ROIRows, ROICols, s).^2 .*W));
                A2 = sum(sum(Lx(ROIRows, ROICols, s).*Ly(ROIRows, ROICols, s).*W));
                A3 = sum(sum(Ly(ROIRows, ROICols, s).^2.*W));
                
                A(:,:,s) = [A1, A2; ...
                            A2, A3];
                        
                b1 = sum(sum((Lx(ROIRows, ROICols, s).^2 .* X + Lx(ROIRows, ROICols, s) .* Ly(ROIRows, ROICols, s) .* Y) .* W));
                b2 = sum(sum((Lx(ROIRows, ROICols, s) .* Ly(ROIRows, ROICols, s) .* X + Ly(ROIRows, ROICols, s).^2 .* Y) .* W ));            
                b(:, s) = [b1; ...
                           b2];
                   
                c = sum(sum((Lx(ROIRows, ROICols, s).^2 .* X.^2 + 2 * Lx(ROIRows, ROICols, s).*Ly(ROIRows, ROICols, s) .* Y .* X + Ly(ROIRows, ROICols, s).^2 .* Y.^2) .* W));
                
                dmin(s) = (c - b(:, s)' * pinv(A(:,:,s)) * b(:, s)) / trace(A(:,:,s));
            end
            % Find minima
            [val, idx] = min(dmin);
                            
            % New estimate of point posititon;
            MHold = MHtmp(counter,:); % save for comparison
            pos = (pinv(A(:,:,idx)) * b(:, idx));
            MHtmp(counter,1:2) = [pos(2), pos(1)];
            
            % Iterate until stopping criterion : 3 iterations, less than 1
            % pixel change or diverge more than 2*sigmaL from original position
            
            % update iterations
            if iter < 4 % Max 3 iterations pr point
                iter = iter + 1;
            else
                iter = 0;
            end
                    
            % less than 1 pixel change
            if norm(MHtmp(counter,1:2) - MHold(1:2)) < 1
                iter = 0;
            end

            % If diverge 
            if norm(MHtmp(counter,1:2) - MH(n,1:2)) > (2 * sigmaL)
                counter = counter - 1;
                iter = 0;
                continue;
            end 
                     
            % If too close to boundary then throw out
            if (MHtmp(counter,1) <= round(2*sigmaL)) | (MHtmp(counter,1) >= (size(I,1)-round(2*sigmaL))) | (MHtmp(counter,2) <= round(2*sigmaL)) | (MHtmp(counter,2) >= (size(I,2)-round(2*sigmaL)))
                counter = counter - 1;
                iter = 0;
                continue;
            end % otherwise keep
                        
        end
       
    end
   
    % Save only valid candidate points
    MH = MHtmp(1:counter,:);
end


MH = [MH(:,1), MH(:,2), k.^(MH(:,3)-1) * sigma0];


end

