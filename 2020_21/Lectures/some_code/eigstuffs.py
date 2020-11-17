#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@project: VIP_2020
    
@filename: eigstuffs.py
    
@description

@author: François Lauze, University of Copenhagen    
Created on Sun Nov  8 16:34:20 2020

"""

import numpy as np
import  matplotlib.pyplot as plt
from math import  cos, sin


__version__ = "0.0.1"
__author__ = "François Lauze"



Pi = np.pi
c30 = cos(Pi/6)
s30 = sin(Pi/6)
R = np.array([])

def normalize(v):
    return v/np.linalg.norm(v)

def rotate(theta, v):
    R = np.array([[cos(theta), -sin(theta)], [sin(theta), cos(theta)]])
    return R@v

def draw_arrow(ax, start, vector, n=None, h=0.2, scale=1.0, **kwargs):
    """
    Draw an vector anchored at start with a head around endpoint
    
    Parameters:
    -----------
    start : numpy float array
        2d coord of start point
    vector : numpy float array
        vector to draw, anchored at start point
    n :  numpy float array
        vector non colinear to stop-start
    h : float 
        real > 0: head size as proportion of length
    scale: float
        scale factor for vector, default 1.0
    """
    
    v = vector*scale
    if n is None:
        n = normalize(rotate(Pi/2., v))
    lv = np.linalg.norm(v)
    vn = v/lv
    fn = n.flatten()
    k = fn - np.dot(vn, fn)*vn
    kn = k/np.linalg.norm(k)
    
    stop = start + v    
    
    P = stop - c30*h*vn + s30*h*kn
    Q = stop - c30*h*vn - s30*h*kn
    
    ax.plot([start[0], stop[0]], [start[1], stop[1]], **kwargs)
    ax.plot([P[0], stop[0]], [P[1], stop[1]],  **kwargs)
    ax.plot([Q[0], stop[0]], [Q[1], stop[1]],  **kwargs)
    

def example_eigenvalues():
    origo = np.zeros((2,))
    v1 = np.array([9.0, -4.0])
    v2 = np.array([-3.0, 1.0])
    v3 = 1.5*v1 - 2.1*v2
    
    A = np.array([[11.0, 27.0], [-4.0, -10.0]])
    fig, ax = plt.subplots(1, 1)
    draw_arrow(ax, origo, v1, color=(1.0,0.0,0.0))
    draw_arrow(ax, origo, v2, color=(0.0,1.0,0.0))
    draw_arrow(ax, origo, v3, color=(0.0,0.0,0.0))
    
    Av1 = A@v1
    Av2 = A@v2
    Av3 = A@v3
    
    draw_arrow(ax, origo, Av1, color=(0.0,1.0,1.0))
    draw_arrow(ax, origo, Av2, color=(1.0,0.0,1.0))
    draw_arrow(ax, origo, Av3, color=(0.0,0.0,0.0))

    # put the name of the vectors v1, v2 and v3 on the plot.
    # around the middle of the correspondiong vector
    # first adjust for the height, so it does not cross
    # some of the arrows
    hv1 = v1/2.0
    hv1[0] -= 1.5
    hv2 = v2/2.0
    hv2[0] -= 1.5
    hv3 = v3/2.0
    
    ax.text(*hv1, r'$v_1$', color=(1.0,0.0,0.0), fontsize=18)
    ax.text(*hv2, r'$v_2$', color=(0.0,1.0,0.0), fontsize=18)
    ax.text(*hv3, r'$v_3$', color=(0.0,0.0,0.0), fontsize=18)
    
    
    
    # do the same thing for their images under A
    hAv1 = Av1/2
    hAv2 = Av2*0.75
    hAv2[0] -= 1.5
    hAv3 = Av3/2.0
    ax.text(*hAv1, r'$Av_1$', color=(0.0,1.0,1.0), fontsize=18)
    ax.text(*hAv2, r'$Av_2$', color=(1.0,0.0,1.0), fontsize=18)
    ax.text(*hAv3, r'$Av_3$', color=(0.0,0.0,0.0), fontsize=18)
    
    
    plt.show()
    
    
    
    
if __name__ == "__main__":
    example_eigenvalues()
    
    
    
    
    