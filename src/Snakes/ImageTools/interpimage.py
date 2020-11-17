# -*- coding: utf-8 -*-
"""
Bilinear and bicubic interpolation of image values.

TO DO: check the bicubic interpolation, there could be a "small" problem
@author: francois
"""

import numpy as np
import math
import finitediffschemes as fd


class InterpImage(object):
    """ Bilinear and bicubic interpolation of values from a 2D function 
    / image. It will be used to interpolate values at curve vertices. 
    I have serious trouble with scipy interpolation routines, so 
    I do it by 'hand'.       
    
    Note that this is done in "matrix" coordinate system. Points 
    obtained from say a plot must be transformed to use ImageInterp.
          
    """
    
    def __init__(self, im):
        
        self.im = im
        self.nx, self.ny = im.shape
        
        # this is used to compute coefficients of bicubic interpolation -- a bit brut force
        # computed and output via Matlab :-)
        self.bcbinverter = np.array([
            [ 1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
            [ 0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
            [-3,  3,  0,  0, -2, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
            [ 2, -2,  0,  0,  1,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
            [ 0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0],
            [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0],
            [ 0,  0,  0,  0,  0,  0,  0,  0, -3,  3,  0,  0, -2, -1,  0,  0],
            [ 0,  0,  0,  0,  0,  0,  0,  0,  2, -2,  0,  0,  1,  1,  0,  0],
            [-3,  0,  3,  0,  0,  0,  0,  0, -2,  0, -1,  0,  0,  0,  0,  0],
            [ 0,  0,  0,  0, -3,  0,  3,  0,  0,  0,  0,  0, -2,  0, -1,  0],
            [ 9, -9, -9,  9,  6,  3, -6, -3,  6, -6,  3, -3,  4,  2,  2,  1],
            [-6,  6,  6, -6, -3, -3,  3,  3, -4,  4, -2,  2, -2, -2, -1, -1],
            [ 2,  0, -2,  0,  0,  0,  0,  0,  1,  0,  1,  0,  0,  0,  0,  0],
            [ 0,  0,  0,  0,  2,  0, -2,  0,  0,  0,  0,  0,  1,  0,  1,  0],
            [-6,  6,  6, -6, -4, -2,  4,  2, -3,  3, -3,  3, -2, -1, -2, -1],
            [ 4, -4, -4,  4,  2,  2, -2, -2,  2, -2,  2, -2,  1,  1,  1,  1],
        ], dtype="float64")


        # I need some derivatives of im for bicubic interpolation
        # and the following computes them with Neumann boundary conditions.        
        self.imx = fd.cdx(im)
        self.imy = fd.cdy(im)
        self.imxy = fd.cdxy(im)
     
     
    def bilinear(self, x, y):
        """ Bilinear interpolation at 1 point. Extrapolation is not
        permitted, and will raise an exception. """
        lx = int(math.floor(x))
        ly = int(math.floor(y))
        px = int(math.ceil(x))
        py = int(math.ceil(y))
        
        if lx < 0 or px >= self.nx:
            raise ValueError('x-value %f is out of bound, should be in [%d,%d]' % (x, 0, self.nx))
        if ly < 0 or py >= self.ny:
            raise ValueError('y-value %f is out of bound, should be in [%d,%d]' % (y, 0, self.ny))
        
        hx = x - lx
        hy = y - ly
        hx1 = 1.0 - hx
        hy1 = 1.0 - hy           
        
        im00 = self.im[lx,ly]
        im10 = self.im[lx+1,ly]
        im01 = self.im[lx,ly+1]
        im11 = self.im[lx+1,ly+1]
        
        return im00*hx1*hy1 + im10*hx*hy1 + im01*hx1*hy + im11*hx*hy
        
        
    def bilinear_list(self, l):
        """Produces a column of bilinearly interpolated values from a 
        list of points. """
        return np.matrix([self.bilinear(l[i,0],l[i,1]) for i in range(len(l))]).T
        
        
    def bicubic(self, x, y):
        """ Bicubic interpolation at 1 point. Extrapolation is not
        permitted, and will raise an exception. """
        lx = int(math.floor(x))
        ly = int(math.floor(y))
        px = int(math.ceil(x))
        py = int(math.ceil(y))
        
        if lx < 0 or px >= self.nx:
            raise ValueError('x-value %f is out of bound, should be in [%d,%d]' % (x, 0, self.nx))
        if ly < 0 or py >= self.ny:
            raise ValueError('y-value %f is out of bound, should be in [%d,%d]' % (y, 0, self.ny))
        
        hx = x - lx
        hy = y - ly
        
        h = np.matrix([hx**i*hy**j for j in range(4) for i in range(4)])
        
        corners = [(0,0), (1,0), (0,1), (1,1)]
        v = np.matrix([im[lx+c[0],ly+c[1]] for im in (self.im, self.imx, self.imy, self.imxy) for c in corners]).T
        a = self.bcbinverter*v;
        ival = h*a
        return ival[0,0]
        
    def bicubic_list(self, l):
        """Produces a column vector of bicubilyly interpolated values from a 
        list of points. """
        return np.matrix([self.bicubic(l[i,0],l[i,1]) for i in range(len(l))]).T
        
        
        
# let's test
if __name__ == "__main__":
    # The following function is bilinear, so apart from boundaries, bilinear and bicubic 
    # interpolations should return the same values.
    a = np.array([[(i+1)*(j+1) + (i+1) - (j+1) for i in range(10)] for j in range(10)])
    
    interpa = InterpImage(a)
    i = 3.65
    j = 7.51
    print "bilinear interpolation at (%f,%f) = " %(i,j), interpa.bilinear(i,j)
    print "bicubic  interpolation at (%f,%f) = " %(i,j), interpa.bicubic(i, j)

    
            