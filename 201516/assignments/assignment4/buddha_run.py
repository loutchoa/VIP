# -*- coding: utf-8 -*-
"""
buddha_run.py

Author: Francois Lauze, University of Copenhagen
Date: Mon Jan  4 15:27:45 2016
"""


import numpy as np
import ps_utils
import numpy.linalg as la

# read Beethoven data
I, mask, S = ps_utils.read_data_file('Buddha')


# get indices of non zero pixels in mask and dimensions
nz = np.where(mask > 0)
m,n,t = I.shape


# In this one I try a normalization trick so that max albedo is 1
nS = la.norm(S, axis=1)
for i in range(t):
    I[:,:,i] /= nS[i]




# for each mask pixel, collect image data
J = np.zeros((t, len(nz[0])))
for i in range(t):
    Ii = I[:,:,i]
    J[i,:] = Ii[nz]


# solve for M = rho*N
pS = la.pinv(S)
M = np.dot(pS, J)

# get albedo as norm of M and normalize M
Rho = la.norm(M, axis=0)
N = M/np.tile(Rho, (3,1))

rho = np.zeros((m,n))
rho[nz] = Rho
ps_utils.display_image(rho)


n1 = np.zeros((m,n))
n2 = np.zeros((m,n))
n3 = np.ones((m,n))
n1[nz] = N[0,:]
n2[nz] = N[1,:]
n3[nz] = N[2,:]

z = ps_utils.simchony_integrate(n1, n2, n3, mask)
ps_utils.display_depth_mayavi(z)


