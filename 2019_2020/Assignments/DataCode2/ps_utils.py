# -*- coding: utf-8 -*-
"""
Project: Vision and Image Processing

File: ps_utils.py

Description: Collections of routines for Photomoetric Stereo 
Processing.

Main content is:

* Two methods of integration of a normal field to a depth function
  by solving a Poisson equation. They are both written for normal field integration,
  i.e., they don't take gradient components, but directly the normal field components.

  - The first, unbiased_integrate(), indeed unbiased, implements a Poisson solver on an 
    irregular domain. Rather standard approach with reflective boundary condiations.
  - The second, simchony_integrate() implements the Simchony et al. method for 
    integration of a normal field.
  
* A specialised implementation of Fishler and Bolles' Random Sampling 
   Consensus -- RANSAC -- for estimating a 3D vector.

* A numerical diffusion/smoothing for normal vector fields (i.e., each 
  vector should have norm 1, the result too) 


François Lauze, University of Copenhagen
Date Started December 2015 / January 2016. Modified August 20167 -- November 2019
"""

import sys
import numpy as np
from scipy import fftpack as fft
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
import random




def ransac_3dvector(data, threshold, max_data_tries=100, max_iters=1000, 
                    p=0.9, det_threshold=1e-1, verbose=2):
    """
    A RANSAC implementation for fitting a vector in a linear model I = s.m
    with I a scalar, s a 3D vector and m the vector to be fitted. 
    For us, s should represent a directional light source, I, observed 
    intensity and m = rho*n = albedo x normal
    Parameters:
    -----------
    data: tuple-like
        pair (I, S) with S the (3,K) matrix of 3D vectors and I the 
        K-vector of observations.
    threshold: float
        |I-s.m| <= threshold: inlier, else outlier.
    max_data_tries: int (optional, default=100)
        maximum tries in sampling 3 vectors from S. If repeatably fail 
        to get a 3x3 submatrix with determinant large enough, then report
        an impossible match here (None instead of a 3D vector as output or exception?)
    max_iters: int (optional, default=1000)
        maximum number of iterations, i.e. explore up to max_iters potential models.
    p: float (optional, default=0.9)
        desired propability to al least pick an essentially error free sample
        (z in Fishler & Bolles paper)
    det_threshold: float
        threshold to decide wether a determinant (in absolute value) is large enough.
    verbose: int
        0 = silent, 1 = some comments, >= 2: a lot of things
    
    Returns:
    --------
    m, inliers, best_fit: tuple (ndarray, int, float) or None
        if successful, m is the estimated 3D vector, inliers a list
        of indices in [0..K-1] of the data points considered as inliers 
        and best_fit the best average fit for the selected model.
        if unsuccessful, returns None

    Reference:
    ----------
    Martin A. Fischler & Robert C. Bolles "Random Sample Consensus: A Paradigm for 
    Model Fitting with Applications to Image Analysis and Automated Cartography". 
    Comm. ACM. 24 (6): 381–395. 1981.
    
    """
    # minimum number of  points to selct a model
    n_model_points = 3
    
    # initialisation of model to None,
    best_m = None
    
    # a count of attempts at selecting a good data set
    trial_count = 0
    
    # score of currently best selected model
    best_score = 0
    
    # best average fit for a model
    best_fit = float("Inf")
    
    # number of trials needed to pick a correct a correct dataset 
    # with probability p. Will be updated during the run (same name 
    # as in Fishler-Bolles.
    k = 1 
    
    I, S = data
    #S = S.T
    S = S.copy().astype(float)
    I = I.copy().astype(float)
    ndata = len(I)
    
    while k > trial_count:
        if verbose >= 2:   
            print("ransac_3dvector(): at trial ",trial_count)

        i = 0
        while i < max_data_tries:
            # select 3 pairs s, I randomely and check whether 
            # they allow to compute a proper model: |det(s_i1, s_i2, s_i3| >> 0.
            idx = random.sample(range(ndata), n_model_points)
            if verbose >= 2:
                print("ransac_3dvector(): selected indices = ", idx)
            s = S[idx]
            if abs(np.linalg.det(s))>= det_threshold:
                Is = I[idx]
                break
            i += 1
        if i == max_data_tries:
            if verbose >= 1:
                print("ransac_3dvector(): no dataset found, degenerate model?")
            return None
        
        # here, we can evaluate a candidate model
        m = np.linalg.inv(s) @ Is
        if verbose >= 2:
            print("ransac_3dvector(): estimated model", m)
        # then its inliers. For that we fist compute fitting values
        fit = np.abs(I - S @ m)
        inliers = np.where(fit <= threshold)[0]
        n_inliers = len(inliers)
        if verbose >= 2:
            print("ransac_3dvector(): number of inliers for this model", n_inliers)

        
        if n_inliers > best_score:
            best_score = n_inliers
            best_inliers = inliers
            # we reevaluate m on the inliers' subset
            s = S[inliers]
            Is = I[inliers]
            best_m = np.linalg.pinv(s) @ Is
            # This should match Yvain's version?
            # best_m = m.copy()
            best_fit = np.mean(np.abs(Is - s@best_m))
            if verbose >= 2:
                print("ransac_3dvector(), updating best model to", best_m)
            
            frac_inliers = n_inliers / ndata
            # p_outliers is the 1 - b of Fishler-Bolles
            # the number of needed points to select a model is 3
            p_outliers = 1 - frac_inliers**n_model_points
            # preveny NaN/Inf  in estimation of k
            eps = np.spacing(p_outliers)
            p_outliers = min(1-eps, max(eps, p_outliers))
            k = np.log(1-p)/np.log(p_outliers)
            if verbose >= 2:
                print("ransac_3dvector(): estimate of runs to select enough inliers with probability {0}: {1}".format(p, k))

            
        trial_count += 1
        if trial_count > max_iters:
            if verbose:
                print("ransac_3dvector(): reached maximum number of trials.")
            break

    if best_m is None:
        if verbose: 
            print("ransac_3dvector(): unable to find a good enough solution.")
        return None
    else:
        if verbose >= 2:
            print("ransac_3dvector(): returning after {0} iterations.".format(trial_count))
        return best_m, best_inliers, best_fit


def cdx(f):
    """
    central differences for f in x direction
    """
    m = f.shape[0]
    west = [0] + list(range(m-1))
    east = list(range(1,m)) + [m-1]
    return 0.5*(f[east,:] - f[west,:])
    
def cdy(f):
    """
    central differences for f in y direction
    """
    n = f.shape[1]
    south = [0] + list(range(n-1))
    north = list(range(1,n)) + [n-1]
    return 0.5*(f[:,north] - f[:,south])
    

#def sub2ind(shape, X, Y):
#    """
#    An equivalent of Matlab sub2ind, but without 
#    argument checking and for dim 2 only.
#    """    
#    Z = np.array(list(zip(X,Y))).T
#    shape = np.array(shape)
#    indices = np.dot(shape, Z)
#    indices.shape = indices.size
#    return indices
    
def tolist(A):
    """
    Linearize array to a 1D list
    """
    return list(np.reshape(A, A.size))
    

def smooth_normal_field(n1, n2, n3, mask, iters=100, tau=0.05, verbose=False):
    """
    Runs a few iterations of a minimization of 2-norm squared
    of a field with domain given by mask and value on the 2-sphere
    Runs dn/dt = -||Grad n||^2, n^Tn = 1

    Arguments:
    ----------
    n1, n2, n3: numpy arrays:
        the x, y and z components of the normal field.
    mask: numpy array
        mask[x,y] == 1 if (x,y) in domain, 0 else.
    iters: int
        number of iterations in descent:
        n^(i+1) = Exp_(S^2,n^(i))(-tau LaplaceBeltrami(n^(i)))
    tau: float
        descent step size
    verbose: bool
        if True, display iteration numbers

    Returns:
    --------
    Smoothed version of field (n1, n2, n3)

    Should I try a Tichonov regularisation instead, i.e. add a reaction toward the
    original normal field?
    """

    #########################################
    # build data used for boundary conditions
    #
    m,n = mask.shape
    inside = np.where(mask)
    x, y = inside
    n_pixels = len(x)
    m2i = -np.ones(mask.shape)
    # m2i[i,j] = -1 if (i,j) not in domain, index of (i,j) else.
    m2i[(x,y)] = range(n_pixels)
    west  = np.zeros(n_pixels, dtype=int)
    north = np.zeros(n_pixels, dtype=int)
    east  = np.zeros(n_pixels, dtype=int)
    south = np.zeros(n_pixels, dtype=int)

    N = np.zeros((n_pixels, 3))
    N[:,0] = n1[x,y]
    N[:,1] = n2[x,y]
    N[:,2] = n3[x,y]

    for i in range(n_pixels):
        xi = x[i]
        yi = y[i]
        wi = x[i] - 1
        ni = y[i] + 1
        ei = x[i] + 1
        si = y[i] - 1

        west[i]  = m2i[wi,yi] if (wi > 0) and (mask[wi, yi] > 0) else i
        north[i] = m2i[xi,ni] if (ni < n) and (mask[xi, ni] > 0) else i
        east[i]  = m2i[ei,yi] if (ei < m) and (mask[ei, yi] > 0) else i
        south[i] = m2i[xi,si] if (si > 0) and (mask[xi, si] > 0) else i


    for i in range(iters):
        if verbose:
            sys.stdout.write('\rsmoothing iteration {0} out of {1}\t'.format(i, iters))
        # Tension (a.k.a vector-valued Laplace Beltrami on proper bundle)
        v3 = N[west] + N[north] + N[east] + N[south] -4.0*N 
        h = (v3*N).sum(axis=-1).reshape((-1,1))
        v = tau*(v3 - h*N)

        # Riemannian Exponential map for evolution

        nv = np.linalg.norm(v, axis=1)
        cv = np.cos(nv).reshape((-1,1))
        sv = np.sin(nv).reshape((-1,1))
        # if  very close to 0, set it to 1, it won't change anything
        np.place(nv, nv < 1e-10, 1.0)
        vhat = v/np.reshape(nv, (-1,1))
        N = cv*N + sv*vhat

    if verbose:
        print('\n')

    N = N.T
    N1 = np.zeros(mask.shape)
    N2 = np.zeros(mask.shape)
    N3 = np.ones(mask.shape)

    N1[inside] = N[0]
    N2[inside] = N[1]
    N3[inside] = N[2]
    return N1, N2, N3




def simchony_integrate(n1, n2, n3, mask):
    """
    Integration of the normal field recovered from observations onto 
    a depth map via Simchony et al. hybrid DCT / finite difference
    methods.
    
    Done by solving via DCT a finite difference equation discretizing
    the equation:
        Laplacian(z) - Divergence((n1/n3, n2/n3)) = 0
    under proper boundary conditions ("natural" boundary conditions on 
    a rectangular domain)
    
    Arguments:
    ----------
    n1, n2, n3: nympy float arrays 
        the 3 components of the normal. They must be 2D arrays
        of size (m,n). The gradient field is computed as (p=-n2/n3, q=-n1/n3)
    mask: characteristic function of domain.
       
    Returns:
    --------
        z : depth map obtained by integration of the field (p, q)

    Reference:
    ----------
    Tal Simchony, Rama Chellappa, and Min Shao. "Direct analytical methods for 
    solving Poisson equations in computer vision problems." IEEE transactions 
    on Pattern Analysis and Machine Intelligence 12.5 (1990): 435-446.
    """
    
    m,n = n1.shape
    p = -n2/n3
    q = -n1/n3

    outside = np.where(mask == 0)
    p[outside] = 0
    q[outside] = 0


    # divergence of (p,q)
    px = cdx(p)
    qy = cdy(q)
    
    f = px + qy      

    # 4 edges
    f[0,1:-1]  = 0.5*(p[0,1:-1] + p[1,1:-1])    
    f[-1,1:-1] = 0.5*(-p[-1,1:-1] - p[-2,1:-1])
    f[1:-1,0]  = 0.5*(q[1:-1,0] + q[1:-1,1])
    f[1:-1,-1] = 0.5*(-q[1:-1,-1] - q[1:-1,-2])

    # 4 corners
    f[ 0, 0] = 0.5*(p[0,0] + p[1,0] + q[0,0] + q[0,1])
    f[-1, 0] = 0.5*(-p[-1,0] - p[-2,0] + q[-1,0] + q[-1,1])
    f[ 0,-1] = 0.5*(p[0,-1] + p[1,-1] - q[0,-1] - q[1,-1])
    f[-1,-1] = 0.5*(-p[-1,-1] - p[-2,-1] -q[-1,-1] -q[-1,-2])

    # cosine transform f (reflective conditions, a la matlab, 
    # might need some check)
    fs = fft.dctn(f, norm='ortho')
    #fs = fft.dct(f, axis=0, norm='ortho')
    #fs = fft.dct(fs, axis=1, norm='ortho')

    # check that this one works in a safer way than Matlab!
    x, y = np.mgrid[0:m,0:n]
    denum = (2*np.cos(np.pi*x/m) - 2) + (2*np.cos(np.pi*y/n) -2)
    Z = fs/denum
    Z[0,0] = 0.0 
    # or what Yvain proposed, it does not really matters
    # Z[0,0] = Z[1,0] + Z[0,1]
    
    z = fft.idctn(Z, norm='ortho')
    #z = fft.dct(Z, type=3, norm='ortho', axis=0)
    #z = fft.dct(z, type=3, norm='ortho', axis=1)
    out = np.where(mask == 0)
    z[out] = np.nan
    return z
# simchony()





def unbiased_integrate(n1, n2, n3, mask, order=2):
    """
    Constructs the finite difference matrix, domain and other information
    for solving the Poisson system and solve it. Port of Yvain's implementation, 
    even  respecting the comments :-)
    
    It creates a matrix A which is a finite difference approximation of 
    the neg-laplacian operator for the domain encoded by the mask, and a
    b matrix which encodes the neg divergence of (n2/n3, n1/n3).
    (Not -n1/n3, -n2/n3, because first dimension in Python / Matlab arrays
    is vertical, second is horizontal, and there are a few weird things with the data)
    
    The depth is obtained by solving the discretized Poisson system
    Az = b, 
    z needs to be reformated/reshaped after that.
    
    Arguments:
    ----------
    n1, n2, n3: nympy float arrays 
        the 3 components of the normal. They must be 2D arrays
        of size (m,n). The gradient field is computed as (p=-n2/n3, q=-n1/n3)
    mask: numpy int/bool array
        characteristic function of domain.
    order: int
        Used to play with discretisations. Don't change, unless you really 
        want too (nothing should explode though...)

    Returns:
    --------
        z : depth map obtained by integration of the field (p, q)

    Reference:
    ----------
    
    """
    
    p = -n2/n3
    q = -n1/n3        
    
    # Calculate some usefuk masks
    m,n = mask.shape
    Omega = np.zeros((m,n,4))
    Omega_padded = np.pad(mask, (1,1), mode='constant', constant_values=0)
    Omega[:,:,0] = Omega_padded[2:,1:-1]*mask
    Omega[:,:,1] = Omega_padded[:-2,1:-1]*mask
    Omega[:,:,2] = Omega_padded[1:-1,2:]*mask
    Omega[:,:,3] = Omega_padded[1:-1,:-2]*mask
    del Omega_padded
    
    # Mapping    
    indices_mask = np.where(mask > 0)
    lidx = len(indices_mask[0])
    mapping_matrix = np.zeros(p.shape, dtype=int)
    mapping_matrix[indices_mask] = list(range(lidx))
    
    if order == 1:
        pbar = p.copy()
        qbar = q.copy()
    elif order == 2:
        pbar = 0.5*(p + p[list(range(1,m)) + [m-1], :])
        qbar = 0.5*(q + q[:, list(range(1,n)) + [n-1]])
        
    # System
    I = []
    J = []
    K = []
    b = np.zeros(lidx)


    # In mask, right neighbor in mask
    rset = Omega[:,:,2]
    X, Y = np.where(rset > 0)
    #indices_center = sub2ind(mask.shape, X, Y)
    I_center = mapping_matrix[(X,Y)].astype(int)
    #indices_neighbors = sub2ind(mask.shape, X, Y+1)
    I_neighbors = mapping_matrix[(X,Y+1)]
    lic = len(X)
    A_center = np.ones(lic)
    A_neighbors = -A_center
    K += tolist(A_center) + tolist(A_neighbors)
    I += tolist(I_center) + tolist(I_center)
    J += tolist(I_center) + tolist(I_neighbors)
    b[I_center] -= qbar[(X,Y)]
    
	
    #	In mask, left neighbor in mask
    lset = Omega[:,:,3]
    X, Y = np.where(lset > 0)
    #indices_center = sub2ind(mask.shape, X, Y)
    I_center = mapping_matrix[(X,Y)].astype(int)
    #indices_neighbors = sub2ind(mask.shape, X, Y-1)
    I_neighbors = mapping_matrix[(X,Y-1)]
    lic = len(X)
    A_center = np.ones(lic)
    A_neighbors = -A_center
    K += tolist(A_center) + tolist(A_neighbors)
    I += tolist(I_center) + tolist(I_center)
    J += tolist(I_center) + tolist(I_neighbors)  
    b[I_center] += qbar[(X,Y-1)]


    # In mask, top neighbor in mask
    tset = Omega[:,:,1]
    X, Y = np.where(tset > 0)
    #indices_center = sub2ind(mask.shape, X, Y)
    I_center = mapping_matrix[(X,Y)].astype(int)
    #indices_neighbors = sub2ind(mask.shape, X-1, Y)
    I_neighbors = mapping_matrix[(X-1,Y)]
    lic = len(X)
    A_center = np.ones(lic)
    A_neighbors = -A_center
    K += tolist(A_center) + tolist(A_neighbors)
    I += tolist(I_center) + tolist(I_center)
    J += tolist(I_center) + tolist(I_neighbors)
    b[I_center] += pbar[(X-1,Y)]


    #	In mask, bottom neighbor in mask
    bset = Omega[:,:,0]
    X, Y = np.where(bset > 0)
    #indices_center = sub2ind(mask.shape, X, Y)
    I_center = mapping_matrix[(X,Y)].astype(int)
    #indices_neighbors = sub2ind(mask.shape, X+1, Y)
    I_neighbors = mapping_matrix[(X+1,Y)]
    lic = len(X)
    A_center = np.ones(lic)
    A_neighbors = -A_center
    K += tolist(A_center) + tolist(A_neighbors)
    I += tolist(I_center) + tolist(I_center)
    J += tolist(I_center) + tolist(I_neighbors)
    b[I_center] -= pbar[(X,Y)]
    
    # Construction de A : 
    A = sp.csc_matrix((K, (I, J)))
    A = A + sp.eye(A.shape[0])*1e-9
    z = np.nan*np.ones(mask.shape)
    z[indices_mask] = spsolve(A, b)
    return z
    


def display_surface(z, albedo=None):
    """
    Display the computed depth function as a surface using 
    mayavi mlab.

    Arguments:
    ----------
    z : numpy array
        2D depth map to be displayed
    albedo: numpy array
        Same size as z if present. Used to render / texture 
        the surface. 
    """
    from mayavi import mlab

    m, n = z.shape
    x, y = np.mgrid[0:m, 0:n]
    x2 = m/2
    y2 = n/2
    if albedo is None:
        scalars = z
    else:
        scalars = albedo.max() - albedo
    mlab.mesh(x, y, z, scalars=scalars, colormap="Greys")
    mlab.view(azimuth=-60, elevation=25.0, focalpoint=(x2, y2,-1.0), distance=2.5*max(m, n))
    mlab.show()
    
    
def display_image(u):
    """
    Display a 2D imag
    """
    from matplotlib import pyplot as plt
    plt.imshow(u)
    plt.show()
    
    
    
def read_data_file(filename):
    """
    Read a matlab PS data file and returns
    - the images as a 3D array of size (m,n,nb_images)
    - the mask as a 2D array of size (m,n) with 
      mask > 0 meaning inside the mask
    - the light matrix S as a (nb_images, 3) matrix

    Arguments:
    ----------
    filename: string
        name of the data file.
    Returns:
    --------
       a triple I, mask, S, where I is the collection  of images,
       size (ny, nx, k), mask is the domain mask (0 out, 1 in), 
       size (ny, nx) and S is the lights vector, size (3, k) (or (k,3)? 
    """
    from scipy.io import loadmat
    
    data = loadmat(filename)
    I = data['I']
    mask = data['mask']
    S = data['S']
    return I, mask, S
    
    