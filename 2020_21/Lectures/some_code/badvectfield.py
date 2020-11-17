# -*- coding: utf-8 -*-
"""""
Project: diffgeo
File: spherestuffs.py

Description: stuffs with parametrized spheres


Author: François Lauze, University of Copenhagen
Date: Sun May 21 20:08:31 2017
"""


from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from math import  cos, sin
Pi = np.pi

N = 20
c30 = cos(Pi/6)
s30 = sin(Pi/6)

def Js2(theta, phi, u):
    Ts2 = np.zeros((3,2))
    Ts2[0,0] = cos(theta)*cos(phi)
    Ts2[0,1] = -sin(theta)*sin(phi)
    Ts2[1,0] = cos(theta)*sin(phi)
    Ts2[1,1] = sin(theta)*cos(phi)
    Ts2[2,0] = -sin(theta)
    Ts2[2,1] = 0.0
    u.shape = (u.size, 1)
    return np.dot(Ts2, u)
    
def param(theta, phi):
    return np.array((sin(theta)*cos(phi), sin(theta)*sin(phi), cos(theta)))
    

def draw_arrow(ax, start, vector, n, h, scale=1.0, **kwargs):
    """
    Draw an vector anchored at start with a head around endpoint
    
    Parameters:
    -----------
    start : numpy float array
        3d coord of start point
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
    lv = np.linalg.norm(v)
    vn = v/lv
    fn = n.flatten()
    k = fn - np.dot(vn, fn)*vn
    kn = k/np.linalg.norm(k)
    
    stop = start + v    
    
    P = stop - c30*h*vn + s30*h*kn
    Q = stop - c30*h*vn - s30*h*kn
    
    ax.plot([start[0], stop[0]], [start[1], stop[1]], [start[2], stop[2]], **kwargs)
    ax.plot([P[0], stop[0]], [P[1], stop[1]], [P[2], stop[2]], **kwargs)
    ax.plot([Q[0], stop[0]], [Q[1], stop[1]], [Q[2], stop[2]], **kwargs)
    

a,b,c = 0.1, -0.2, 1.1
d,e,f = -0.1,0.3,-1.4
t = np.linspace(0.1,1.9)
thetat = a + b*t + c*t**2
phit = d + e*t + f*t**2


theta, phi = np.mgrid[0:Pi:N*1j, 0:2*Pi:2*N*1j]
x = np.sin(theta)*np.cos(phi)
y = np.sin(theta)*np.sin(phi)
z = np.cos(theta)


fig = plt.figure()

ax3 = fig.add_subplot(111, projection='3d')
ax3.plot_wireframe(x,y,z, alpha=0.4)
ax3.axis('off')


xs = np.sin(thetat)*np.cos(phit)
ys = np.sin(thetat)*np.sin(phit)
zs = np.cos(thetat)

ax3.plot(xs, ys, zs, color=[1,0,0], linewidth=2)

thetapt = b + 2*c*t
phipt = e + 2*f*t

thetappt = 2*c
phippt = 2*f


xsp = thetapt*np.cos(thetat)*np.cos(phit) - phipt*np.sin(thetat)*np.sin(phit)
ysp = thetapt*np.cos(thetat)*np.sin(phit) + phipt*np.sin(thetat)*np.cos(phit)
zsp = -thetapt*np.sin(thetat)


xspp = thetappt*np.cos(thetat)*np.cos(phit) - (thetapt**2 + phipt**2)*np.sin(thetat)*np.cos(phit) - 2*thetapt*phipt*np.cos(thetat)*np.sin(phit)
yspp = thetappt*np.cos(thetat)*np.sin(phit) - (thetapt**2 + phipt**2)*np.sin(thetat)*np.cos(phit) + 2*thetapt*phipt*np.cos(thetat)*np.cos(phit)
zspp = -thetappt*np.cos(thetat) + thetapt**2*np.sin(thetat)




# draw curve
ax3.plot(xs, ys, zs, color=[1,0,0], linewidth=2)
for i in range(1,len(xsp),4):

    # Anchor point omn the curve    
    start = np.array([xs[i],ys[i],zs[i]])

    # draw velocity field, scaled down
    vector = np.array([xsp[i], ysp[i], zsp[i]])
    draw_arrow(ax3, start, vector, np.array([1.0,1.0,1.0]), 0.1, scale=0.04, color=[0,.75,0], linewidth=1.5)
    
    # draw "naive" acceleration
    vector = np.array([xspp[i], yspp[i], zspp[i]])
    draw_arrow(ax3, start, vector, np.array([1.0,1.0,1.0]), 0.1, scale=0.025, color=[0,0,0], linewidth=1.5)
    
    # draw covariant acceleration
    vector = np.array([xspp[i], yspp[i], zspp[i]])
    vector = (vector - np.dot(vector, start)*start)
    draw_arrow(ax3, start, vector, np.array([1.0,1.0,1.0]), 0.1, scale=0.025, color=[0,0,0.75], linewidth=1.5)
    

plt.show()