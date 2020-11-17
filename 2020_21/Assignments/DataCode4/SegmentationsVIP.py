#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File SegmentationsVIP.py

@Project: VIP2020

@Description: Simple implementations of Otsu thresholding
algorithm and 2-means clustering (should give similar results)

@Author Francois Lauze, University of Copemhagen (francois@di.ku.dk)
@Date 17/6/2017-1/5/2020

"""
__version = "0.0.1"

import numpy as np
import matplotlib.pyplot as plt
from imageio import imsave, imread
from skimage.data import page, camera, coins
import random


def otsu(img):
    """
    Computes histogram threshold for segmentation using
    Otsu's algorithm, which maximises interclass variance
    from histogram.

    Parameters:
    img: ndarray
        a 2D ndarray representing an image.
    
    Returns:
        t: intensity threshold. so that img <= t represents 1 class
        and img > t represents the other one. 
    """

    min_int = img.min()
    max_int = img.max()
    bins = max_int - min_int + 1
    rg_int = np.arange(min_int, max_int + 1)

    # np.histogram returns the histogram and bin edges,
    # they are unused here.
    h, _ = np.histogram(img, bins)
    
    # ih will allow tp speed up means computations
    ih = h*rg_int    
    nb_pixels = h.sum()
    sum_vals = ih.sum()

    # _, ax = plt.subplots(1,1)
    # ax.bar(rg_int, h)
    # plt.show()

    n1 = 0
    n2 = 0
    max_var = 0.0
    t = rg_int[0]
    ts = t
    m1 = 0.0
    m2 = sum_vals
    vars = [0]

    for t in rg_int[:-1]:
        n1 += h[t-min_int]
        n2 = nb_pixels - n1
        m1 += ih[t-min_int]
        m2 -= ih[t-min_int]

        omega1 = n1/nb_pixels
        omega2 = n2/nb_pixels
        mu1 = m1/n1
        mu2 = m2/n2
        var = omega1*omega2*((mu1-mu2)**2)
        vars.append(var)
        if var > max_var:
            max_var = var
            ts = t  

    # _, ax = plt.subplots(1,1)
    # ax.plot(np.array(vars))
    # plt.show()

    return ts
    


def two_means(img, maxiters=100):
    """
    Segmentation by running a k-means, k=2, minimisation
    using Lloyd's algorithm.
    """

    # initialise with 2 random intensities
    t1, t2 = random.sample(range(img.min(), img.max()+1), 2)
    
    for iter in range(maxiters):
        # compute square distances to cluster centers
        d1 = (img - t1)**2
        d2 = (img - t2)**2

        # cluster values
        idx1 = np.where(d1 <= d2)
        idx2 = np.where(d1 > d2)

        # update centers
        t1 = np.mean(img[idx1])
        t2 = np.mean(img[idx2])

    # return the clusters
    s1 = np.zeros_like(img)
    s2 = np.zeros_like(img)
    s1[idx1] = 1
    s2[idx2] = 1
    return s1, s2


def k_means(img, k, maxiters=100):
    """
    Segmentation by running a more general k-means,
    minimisation using Lloyd's algorithm.
    """
    # initialise with k random intensities
    t = random.sample(range(img.min(), img.max() + 1), k) 
    t = np.array(t)
   
    # redim to vectorise calculation
    omg = np.reshape(img, img.shape + (1,))
    
    for iter in range(maxiters):
        t.shape = (1,1,k)
        # compute square distance to cluster centers
        d2 = (omg - t)**2

        # cluster values
        c = np.argmin(d2, axis=-1)

        # recompute cluster centers (i.e., t)
        # maybe I can vectorise but...
        t.shape = k
        for i in range(k):
            x, y = np.where(c == i)
            if len(x) == 0:
                print("Warning: cluster {0} is empty.".format(i))
            t[i] = np.mean(img[x, y])

    return c



def smooth_segmentation(img, neighborhood=4, majority=None, iters=1):
    """
    Runs a simple 'hole' filler algorithm by 'majority filling'
    Parameters:
    -----------
    img: ndarray
        a binary image representing a segmentation.
    neighborhood: int
        4 for 4-neighbors system, 8 for 8-neighbors system
    majority: int or None
        Replace center value by dominant neighbors ones if 
        there are al least 'majority' identical values in
        a neighborhood. If None, set to 4 for 4-neighbors 
        system, 7 for 8 neighbor systems. 

    """
    if majority is None:
        majority = 4 if neighborhood == 4 else 7
        
    c4 = np.array([
        [0, 1, 0],
        [1, 0, 1],
        [0, 1, 0]
    ])
    c8 = np.array([
        [1, 1, 1],
        [1, 0, 1],
        [1, 1, 1]
    ])
    m, n = img.shape
    img_c = img.copy()
    cleaned = img.copy()
    neighb = c4 if neighborhood == 4 else c8

    res = []

    for iter in range(iters):
        # we do not iterate along image boundaries, simpler.
        for i in range(1,m-1):
            for j in range(1,n-1):
                # count the number of 1s around pixel (i,j)
                nij = img_c[i-1:i+2,j-1:j+2]
                ones = (nij*neighb).sum()
                if ones >= majority:
                    cleaned[i, j] = 1
                else:
                    zeros = neighborhood - ones
                    if zeros > majority:
                        cleaned[i, j] = 0
        res.append(cleaned)
        img_c = cleaned
        cleaned = img_c.copy()

    return res


if __name__ == "__main__":
    #img = page()
    #img = camera()
    #img = coins()
    img = imread('WIG1T_0472_small.png')
    t = otsu(img)
    s1 = (img <= t).astype('uint8')
    s2 = (img > t).astype('uint8')

    f, (ax0, ax1, ax2) = plt.subplots(1, 3, sharex=True, sharey=True)
    ax0.imshow(img, cmap='Greys_r')
    ax1.imshow(s1, cmap='Greys_r')
    ax2.imshow(s2, cmap='Greys_r')
    plt.suptitle("Otsu thresholding with best threshold = {0}".format(t))
    #plt.show()

    s1, s2 = two_means(img, maxiters=200)
    f, (ax0, ax1, ax2) = plt.subplots(1, 3, sharex=True, sharey=True)
    ax0.imshow(img, cmap='Greys_r')
    ax1.imshow(s1, cmap='Greys_r')
    ax2.imshow(s2, cmap='Greys_r')
    plt.suptitle("2-means thresholding")
    plt.show()  

    res = smooth_segmentation(s1, neighborhood=8, majority=4, iters=5)
    f, ax = plt.subplots(2, 3, sharex=True, sharey=True)
    f.canvas.set_window_title("Segmentation cleaning.")
    ax0 = ax[0,0]
    ax0.imshow(s1, cmap='Greys_r')
    ax0.set_title('before cleaning')
    bx = ax.flatten()[1:]
    for i in range(5):
        bx[i].imshow(res[i], cmap='Greys_r')
        bx[i].set_title('after {0} cleaning iteration{1}'.format(i+1, 's' if i > 0 else ''))
    plt.show()


    segs = []
    for k in [2,3,4,5,6,7,8,9,10,11,12]:
        segs.append(k_means(img, k))
    
    f, ax = plt.subplots(3,4)
    f.canvas.set_window_title("Different ks.")
    ax0 = ax[0,0]
    ax0.imshow(img, cmap='Greys_r')
    ax0.set_title('Input image')
    bx = ax.flatten()[1:]
    for i in range(len(bx)):
        bx[i].imshow(segs[i])
        bx[i].set_title('{0}-means'.format(i+2))
    plt.show()

