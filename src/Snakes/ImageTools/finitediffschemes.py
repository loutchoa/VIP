# -*- coding: utf-8 -*-
"""
Small order finitedifference schemes for images
using Neumann boundary conditions.
This means that I do switch between x and y axes!
@author: francois
"""

import numpy as np



def nrange(n):
    """ Shift the index to the left, the last index being repeated. 
    Used to implement Neumann boundary conditions."""
    return range(1,n) + [n-1]


def prange(n):
    """ Shift the index to the right, the first index being repeated. 
    Used to implement Neumann boundary conditions."""
    return [0] + range(0,n-1)


# Central difference schemes.





def cdx(im):
    """ Compute derivatives of im in horizontal direction
    using a central difference scheme with Neumann boundary conditions. 
    The grid step size is not included. """
    nx, ny = im.shape
    nnx = nrange(nx)
    pnx = prange(nx)
    return (im[nnx,:] - im[pnx,:])/2.0
    
    
def cdy(im):
    """ Compute derivatives of im in vertical direction
    using a central difference scheme with Neumann boundary conditions. 
    The grid step size is not included. """
    nx, ny = im.shape
    nny = nrange(ny)
    pny = prange(ny)
    return (im[:,nny] - im[:,pny])/2.0
    
    
def cdxy(im):
    """Compute mixed derivatives using a central differences 
    scheme with Neumann boundary conditions. """
    return cdx(cdy(im))


def cdxx(im):
    """ Compute second order derivatives in x using a central differences 
    scheme with Neumann boundary conditions."""
    nx, ny = im.shape
    nnx = nrange(nx)
    pnx = prange(nx)
    return (im[nnx,:] -2*im + im[pnx,:])/2.0
    

def cdyy(im):
    """ Compute second order derivatives in y using a central differences 
    scheme with Neumann boundary conditions."""
    nx, ny = im.shape
    nny = nrange(ny)
    pny = prange(ny)
    return (im[:,nny] -2*im + im[:,pny])/2.0
    

# Forward difference schemes. 1st order only

def fdfx(im):
    """ Compute the first order derivative in x with a forward differences 
    scheme with Neumann boundary conditions. """
    nx, ny = im.shape
    nnx = nrange(nx)
    return im[nnx:,] - im
     
    
def fdfy(im):
    """ Compute the first order derivative in x with a forward differences 
    scheme with Neumann boundary conditions. """
    nx, ny = im.shape
    nny = nrange(ny)
    return im[:,nny] - im   
    
    
# Backward difference schemes. 1st order only

def bdfx(im):
    """ Compute the first order derivative in x with a forward differences 
    scheme with Neumann boundary conditions. """
    nx, ny = im.shape
    pnx = prange(nx)
    return im - im[pnx,:]
     
    
def bdfy(im):
    """ Compute the first order derivative in x with a forward differences 
    scheme with Neumann boundary conditions. """
    nx, ny = im.shape
    pny = prange(ny)
    return im - im[:,pny]   
        
    
    
    
    
if __name__ == "__main__":
    a = np.array([[(i+1)*(j+1) for i in range(10)] for j in range(10)])
    ax = cdx(a)
    ay = cdy(a)
    axy = cdxy(a)
    ayx = cdy(cdx(a))
    axx = cdxx(a)
    ayy = cdyy(a)
    
    

