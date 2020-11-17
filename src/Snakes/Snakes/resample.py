# -*- coding: utf-8 -*-
"""
Resampling of sampled curve. 
@author: francois
"""


import numpy as np
from scipy import interpolate as ip




def resample(v):
    """ This function takes the vertices of a curve and produce a new series
    that is close enough to the old ones while being better equisampled. 
    This is a cheap approximation to the bicycle chain model that we proposed
    with Stefan and Aditya.    
    """
    # number of vertices
    av = np.array(v)    
    n = av.shape[0]
    dv = av - np.roll(av,-1,axis=0)
    # total Euclidean length of the curve 
     
    # I compute the length of each segment and the total 
    # length of the curve
    cumseglenv = (np.sqrt((dv**2).sum(axis=1))).cumsum()
    t_old = np.hstack(([0], cumseglenv))
    x_old = np.hstack((av[:,0], [av[0,0]]))
    y_old = np.hstack((av[:,1], [av[0,1]]))    
    interpx = ip.interp1d(t_old, x_old, kind='linear')
    interpy = ip.interp1d(t_old, y_old, kind='linear')
    
    # cumseglenv[end] contains the total length of the curve.
    step = cumseglenv[-1]/n
    t_new = np.linspace(0,n*step,n)
    x_new = interpx(t_new)
    y_new = interpy(t_new)
    v_new = np.vstack((x_new,y_new)).T
    return np.matrix(v_new)
    
