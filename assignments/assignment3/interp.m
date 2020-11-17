function Ipim = interp(startim, point, values, maxitt, lambda)
% TPS-approximation of sparse measurements
    
% This matlab-function is a readable, but not very efficient,
% example on how a thin plate splide approximation of sparse
% measurements may be achieved.  
% 
% The procedure is itetative and runs for maxitt iterations. The
% images 'startim', 'points' and 'values' should all be of the same
% size (this is not checked). The image 'point' is binary/logical
% and has a value of 1 in the pixels where there is a
% measurement. The double valued measure is stored in 'values'.
% The image 'startim' is used as initial value for the iterative
% updating.  It may just be zero, but a better value (eg. the
% average measurement value or some other prior) may speed up 
% the convergence. Tha parameter 'lambda' is the regularization
% parameter and determines the smoothness of the result.
    
% Please notice that border values are treated very simple. There
% is no interpolation breaks at discontinuities, and there is no
% monitoring of convergence progress etc. 
   
    
% Please feel free to use and modify, but don't complain 
% Søren I. Olsen, December 2014
    
    
    itt  = 0;
    dimx = size(startim,1);
    dimy = size(startim,2);
    Ipim = startim;
    temp = startim;
    while (itt < maxitt)
        for x = 2:dimx-1
            for y = 2:dimy-1
                av = Ipim(x-1,y) + Ipim(x+1,y) + Ipim(x,y-1) + Ipim(x,y+1);
                if (point(x,y) == 1)
                    temp(x,y) = (values(x,y) + lambda*av)/(1.0 + 4*lambda);
                else
                    temp(x,y) = av/4.0;
                end
            end
        end
        % copy values to borders
        for y = 1:dimy
            temp(1,y) = temp(3,y);
            temp(dimx,y) = temp(dimx-2,y);
        end
        for x = 1:dimx
            temp(x,1) = temp(x,3);
            temp(x,dimy) = temp(x,dimy-2);
        end
            
        Ipim = temp;
        itt = itt +1;
    end  
end  %---interp


