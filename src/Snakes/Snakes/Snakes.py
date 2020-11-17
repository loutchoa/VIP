# -*- coding: utf-8 -*-
"""
Snake1 implements the Kass-Witkin-Terzopoulos active
contours a.k.a Snakes with a specific demo-kind of interaction.
@author: francois
"""


import sys
sys.path.append('../ImageTools')

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage.filters import gaussian_filter as gf
from scipy import interpolate
import interpimage as ip
import finitediffschemes as fd
from resample import resample
import math

class Snake(object):
    """ Class that contains Snake parameters and more comments
    to follow later when appropriate. """
    
    def __init__(self, im, alpha, beta, gamma, delta, tau, n, resample=0):
        """ Init the algorithm with:
            image to segment: im            
            weight parameters:
                - alpha: curve first derivative ( for length / curvature)
                - beta: curve stiffness (second derivative)
                - gamma: external forces weight
                - delta: ballon forces weight
            descent parameter:
                - tau
            other parameter:
                - n : number of points of the snake curve
                - resample: interger m, default is 0. If resample > 0,
                  then the snake is regularly, every m iterations.
               
        """
                
        self.im = Snake.Normalize(im)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.tau = tau
        self.n = n
        self.M = self.SystemMatrix()
        self.resample = resample        
        
    @staticmethod        
    def Normalize(im):
        """ Normalize an image to range [0,1]. Makes my life easier to 
        set the snake parameters. """
        immax = im.max()
        immin = im.min()
        irange = immax - immin
        return (im-immin)/irange
        
        
        
    def SystemMatrix(self):        
        """ Compute the system matrix of the snake evolution, without
        balloon forces (I should add them). """
        
        # line represents the first line of the matrix. All the others
        # are gotten by circular permutation.
        line = np.zeros((self.n))
        line[0] = 1+ self.tau*(2*self.alpha + 6*self.beta)
        line[1] = -self.tau*(self.alpha + 4*self.beta)
        line[2] = self.tau*self.beta
        line[-1] = line[1]
        line[-2] = line[2]
        D = line
        for i in range(1,self.n):
            D = np.vstack((D,np.roll(line,i)))
    
        return np.linalg.inv(D)
    
        
        
    def SetExtforces(self, ef=None, efx=None, efy=None, interp = 'bicubic'):
        """ Set the external forces. If ef is provided, its gradient is computed.
        Else efx and efy must be provided as gradient components. """
        if ef:
            self.efx = fd.cdx(ef)
            self.efy = fd.cdy(ef)
        else:
            self.efx = efx
            self.efy = efy
        self.ipefx = ip.InterpImage(efx)
        self.ipefy = ip.InterpImage(efy)
        self.interp = interp
                    
                    
    def SetExtforcesCanny(self,sigma=3.0, interp='bicubic'):
        """ This sets external forces à la Canny,
        that is -1/2||Grad I_\sigma||^2. """
        imsx  = gf(self.im, sigma, (1,0))
        imsy  = gf(self.im, sigma, (0,1))
        imsxx = gf(self.im, sigma, (2,0))
        imsxy = gf(self.im, sigma, (1,1))
        imsyy = gf(self.im, sigma, (0,2))
        efx = -(imsx*imsxx + imsy*imsxy)
        efy = -(imsx*imsxy + imsy*imsyy)
        self.ef = -(imsx**2 + imsy**2)
        self.SetExtforces(efx=efx, efy=efy, interp=interp)
        
        
    def SetExtforcesRatio(self, sigma=3.0, eps=1, interp='bicubic'):
        """ As a variation of extforce_kwt, set external forces to
        1/(||Grad I_\sigma||^2 + eps). """
        imsx = gf(self.im, sigma, (1,0))
        imsy  = gf(self.im, sigma, (0,1))
        imsxx = gf(self.im, sigma, (2,0))
        imsxy = gf(self.im, sigma, (1,1))
        imsyy = gf(self.im, sigma, (0,2))
        gs2 = imsx**2 + imsy**2 + eps
        efx = -(imsx*imsxx + imsy*imsxy)/gs2
        efy = -(imsx*imsxy + imsy*imsyy)/gs2
        self.ef = 1.0/np.sqrt(imsx**2 + imsy**2 + eps)
        self.SetExtforces(efx=efx, efy=efy, interp=interp)


    def SetExtforcesMarrHildredth(self, sigma=3.0, interp='bicubic'):
        """ Set external forces to a Marr-Hildreth type. 
        -(Laplacian I_\sigma)^2. """
        imsxx  = gf(self.im, sigma, (2,0))
        imsyy  = gf(self.im, sigma, (0,2))
        imsxxx = gf(self.im, sigma, (3,0))
        imsxxy = gf(self.im, sigma, (2,1))
        imsxyy = gf(self.im, sigma, (1,2))
        imsyyy = gf(self.im, sigma, (0,3))
        lapims = imsxx + imsyy
        efx = -lapims*(imsxxx + imsxxy)
        efy = -lapims*(imsxyy + imsyyy)
        self.ef = -0.5*lapims**2
        self.SetExtforces(efx=efx, efy=efy, interp=interp) 
        
        
    def SetExtforcesPeronaMalik(self, sigma=3.0, w=1.0, interp='bicubic'):
        """ Set the external forces to a Perona-Malik type edge term:
        exp(-||Grag I_\sigma||^2/w). """
        ims = gf(self.im, sigma)
        imsx  = fd.cdx(ims)
        imsy  = fd.cdy(ims)
        imsxx = fd.cdxx(ims)
        imsxy = fd.cdxy(ims)
        imsyy = fd.cdyy(ims)
        gs2 = imsx**2 + imsy**2
        self.ef = -np.exp(gs2/w)
        efx = -(2.0/w)*(imsx*imsxx + imsy*imsxy)*self.ef
        efy = -(2.0/w)*(imsy*imsxy + imsy*imsyy)*self.ef
        self.SetExtforces(efx=efx, efy=efy, interp=interp)
        
        
    def ExtForces(self, v):
        fx = self.ipefx.bilinear_list(v)
        fy = self.ipefy.bilinear_list(v)
        return np.hstack((fx,fy))

         
        
    def BalloonForces(self, v):
        # I assume that I have the correcy contour orientation...
        dv = np.array(np.roll(v,-2,axis=0) - np.roll(v,2,axis=0))
        ndv = np.sqrt((dv**2).sum(axis=1))
        nx = dv[:,0]
        ny = dv[:,1]
        nnx = nx/ndv
        nny = ny/ndv
        return np.matrix(np.vstack((-nny,nnx)).T)

        
    def Iteration(self, v, iters):
        """ A generator which produces iterations of the snake algorithm. 
        WHen the generator is created, v should be initialized with
        staring values of the snake. 
        In fact, at each call of next(), the current value of the curve is returned
        as well as the displacement field. The updated value of the curve will be
        returned at the next iteration. This is done in order to show the forces
        acting on the snake at a given iteration.
        """
        vn = v        
        for i in range(iters):
            v = vn
            if self.resample > 0 and i % self.resample == 0: 
                v = resample(v)
       
                
            vn = self.M*(v - self.tau*self.gamma*self.ExtForces(v) + self.tau*self.delta*self.BalloonForces(v))
            yield v, vn-v
                
            
            
    def CreateGenerator(self, v, iters):
        """ Creates the generator object that will iterate the snake.
        """
        if not hasattr(self, 'interp'):
            # if interp does not exists, external forces were not set. Default to 
            # Canny ones with default standard deviation.
            self.SetExtforcesCanny()
        return self.Iteration(v, iters)
        
# end class Snake







