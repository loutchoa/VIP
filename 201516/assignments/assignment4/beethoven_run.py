# -*- coding: utf-8 -*-
"""
Run Beethoven reconstruction code 

Author: Francois Lauze, University of Copenhagen
Date: Mon Jan  4 14:11:54 2016
"""


import numpy as np
import ps_utils
import numpy.linalg as la

# read Beethoven data
I, mask, S = ps_utils.read_data_file('Beethoven')

# get indices of non zero pixels in mask
nz = np.where(mask > 0)
m,n = mask.shape

# for each mask pixel, collect image data
J = np.zeros((3, len(nz[0])))
for i in range(3):
    Ii = I[:,:,i]
    J[i,:] = Ii[nz]


# solve for M = rho*N
#iS = la.inv(S)
#M = np.dot(iS, J)
# alternatively, I can use
M = la.solve(S,J)
# or 
#M = la.lstsq(S,J)



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
#outmask=np.where(mask==0)
#z[outmask] = 0.0
ps_utils.display_depth_mayavi(np.fliplr(z))


