#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Project: Vision and Image Processing

File: ps-solver.py
    
description:
A (quick and not totally dirty) generic solution of PS problems for 
the assignment. Fully rewritten from past years, with
RANSAC and normal fields smoothing.

Francois Lauze, University of Copenhagen
2019-2020
"""
import sys
import numpy as np
# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
from matplotlib import pyplot as plt
import ps_utils
# from scipy.io import savemat

all_datasets = {
    'Beethoven': ['Beethoven.mat', 1.0],
    'mate_vase': ['mat_vase.mat', 2.0],
    'shiny_vase': ['shiny_vase.mat', 2.0],
    'shiny_vase2': ['shiny_vase2.mat', 2.0],
    'face': ['face.mat', 10.0],
    'Buddha': ['Buddha2.mat', 30.0]
}


def load_dataset(dataset):
    try:
        filename, thresh = all_datasets[dataset]
        I, mask, S = ps_utils.read_data_file(filename)
    except Exception as inst:
        print(inst)
        print(dataset, ' not found. exiting.')
        sys.exit(1)
    return I, mask, S, thresh


def fig_grid(n_images):
    """ computes a grid size for displaying images and light sources"""
    cols = int(np.ceil(np.sqrt(n_images)))
    lines = cols - 1 if cols ** 2 > n_images else cols
    if lines * cols < n_images:
        lines += 1
    return lines, cols


def display_images_and_light(I, S):
    """ Display all the images in a data set
        and the corresponding light vectors
    """
    n_images = I.shape[-1] + 1
    lines, cols = fig_grid(n_images)
    fig = plt.figure()

    for k in range(n_images):
        if k < n_images - 1:
            # Plot images in the dataset
            ax = fig.add_subplot(lines, cols, k + 1)
            ax.imshow(I[:, :, k], cmap='Greys_r')
            ax.set_title('Image {0}'.format(k + 1))
            ax.axis('off')
            k += 1
        if k == n_images - 1:
            # Plot light sources directions
            ax = fig.add_subplot(lines, cols, k + 1, projection='3d')
            l = S.shape[0]
            mS = np.linalg.norm(S, axis=-1).max() * 15.0
            T = S/mS
            ax.quiver(np.zeros(l), np.zeros(l), np.zeros(l), T[:, 0], T[:, 1], T[:, 2])
            # ax.quiver(np.zeros(l), np.zeros(l), np.zeros(l), T[:, 0] / mS, S[:, 1] / mS, S[:, 2] / mS)
            ax.set_title('Light sources')
            # ax.axis('off')
            # ax.axis('equal')
    plt.show()



def recons(I, mask, S, perm=None):
    """ reconstruction with permutation of light vectors, as
        something is wrong with Beethoven!
        perm should be an n_lights long array of unique indices in [0..n_lighs-1]
        set perm to None to do nothing
    """
    inside = np.where(mask > 0)
    # outside = np.where(mask == 0)

    npixels = len(inside[0])
    n_lights = S.shape[0]

    # applies light permutation
    T = S.copy()
    if perm is not None:
        for i, idx in enumerate(perm):
            T[i] = S[idx]

    J = np.zeros((n_lights, npixels))
    for i in range(n_lights):
        J[i] = I[:, :, i][inside]

    # solve for tha albedo weighted normal field
    M = np.linalg.lstsq(T, J)[0]
    rho = np.linalg.norm(M, axis=0)
    M /= np.reshape(rho, (1, -1))
    # albedo between 0 and 1
    rho /= rho.max()

    # create a 2D image for the albedo
    # 2D fields for each component of the normal vector field
    rho2D = np.zeros_like(mask, dtype=float)
    n1 = np.zeros_like(mask, dtype=float)
    n2 = np.zeros_like(mask, dtype=float)
    n3 = np.zeros_like(mask, dtype=float)
    n1[inside] = M[0]
    n2[inside] = M[1]
    n3[inside] = M[2]
    # integrate by solving the Poisson equation
    z = ps_utils.unbiased_integrate(n1, n2, n3, mask)
    # dirty trick for maybe fixing convex/concave ambiguity
    if z[0, 0] > z[z.shape[0] // 2, z.shape[1] // 2]:
        z = -z
    return M, rho, z

def standard_reconstruction(I, mask, S):
    """ 
    Woodham-type reconstruction without applying RANSAC 
    First the albedo-modulated normal field is computed,
    Its norm provides the albedo (after normalisation)
    Its normalisation provides teh surface normal field
    and we integrate it to get a surface.
    """

    inside = np.where(mask > 0)
    # outside = np.where(mask == 0)

    npixels = len(inside[0])
    n_lights = S.shape[0]

    # M will contained the "vectorized" normal vector field
    # M = np.zeros((3, npixels))

    # flatten things to go fast
    # I want S (K, 3), I(K, M) with M number of pixels in mask
    J = np.zeros((n_lights, npixels))
    # print('number of lights', n_lights)
    for i in range(n_lights):
        J[i] = I[:, :, i][inside]
    # in lectures I talked about pseudo-inverse, and only a bit about least-squares
    pS = np.linalg.pinv(S)
    # solve for the vector field rho*normal
    M = pS @ J
    #rho = np.sqrt((M ** 2).sum(axis=0))
    rho = np.linalg.norm(M, axis=0)
    M /= np.reshape(rho, (1, -1))
    # "abstract" albedo should be in [0,1], so just normalise
    rho /= rho.max()

    # (re)create 2D fields
    # scalar albedo field
    rho2D = np.zeros(mask.shape)
    # normal fields
    rho2D[inside] = rho
    plt.imshow(rho2D, cmap='Greys_r')
    plt.axis('off')
    plt.title('Computed albedo')
    plt.show()
    n1 = np.zeros(mask.shape)
    n2 = np.zeros(mask.shape)
    n3 = np.ones(mask.shape)

    n1[inside] = M[0]
    n2[inside] = M[1]
    n3[inside] = M[2]
    f, (axn1, axn2, axn3) = plt.subplots(1, 3)
    axn1.imshow(n1)
    axn1.set_title('y-component of normal field.')
    axn2.imshow(n2)
    axn2.set_title('x-component of normal field.')
    axn3.imshow(n3)
    axn3.set_title('z-component of normal field.')
    for ax in axn1, axn2, axn3:
        ax.axis('off')
    plt.show()

    z = ps_utils.unbiased_integrate(n1, n2, n3, mask)
    # If I want to use the alternate integration method (Fourier based)
    # z = ps_utils.simchony_integrate(n1, n2, n3, mask)
    # simple dirty trick to solve convex / concave ambiguity (Yvain). 
    # Sometimes works
    if z[0, 0] > z[z.shape[0] // 2, z.shape[1] // 2]:
        z = -z
    ps_utils.display_surface(z, rho2D)
    # if Mayavi does not work: comment the line above
    # uncomment line below (slow!)
    # ps_utils.display_surface_matplotlib(z, rho2D)

    return M, rho, z

def reconstruction_with_ransac(I, mask, S, thresh):
    """ Incorporate a RANSAC filtering to the Woodham reconstruction. """
    inside = np.where(mask > 0)
    # outside = np.where(mask == 0)

    npixels = len(inside[0])
    n_lights = S.shape[0]

    if n_lights == 3:
        print("Just 3 light sources: RANSAC will provide the same solution as Woodham!")

    M = np.zeros((3, npixels))
    rho2D = np.zeros(mask.shape)

    verbose = 0  # Set to > 0 when desperate, should help tracking bugs...
    for i in range(npixels):
        res = None
        if verbose == 0:
            sys.stdout.write('\rpixel {0} out of {1}     '.format(i, npixels - 1))
        else:
            print('Processing pixel ', i)
        u = inside[0][i]
        v = inside[1][i]

        Ii = I[u, v]
        res = ps_utils.ransac_3dvector((Ii, S), thresh, p=0.99, verbose=verbose, det_threshold=1e-2)

        # I may return a null vector as estimate of the normal because
        # I am actually just outside the mask or have seriously weird measurements?
        # So if RANSAC does not fails but returns a null vector, set it to (0,0,1)
        # Remark:
        # - I could set it to (0,0,very_small?),
        # - I could mark it as invalid and remove the corresponding location from the mask
        # - or I could mark as "to be inpainted.... (seems to be the best solution to me)"
        if (res is not None) and (np.linalg.norm(res[0]) > 1e-5):
            M[:, i] = res[0]
        else:
            M[:, i] = (0.0, 0.0, 1.0)

    print('\n')
    entry, idx = np.where(np.abs(M) == float('inf'))
    if entry.size > 0:
        print("Ups: got some infinite values...")
    print("Max M = ", M.max())
    print("Min M = ", M.min())

    rho = np.sqrt((M ** 2).sum(axis=0))
    M /= np.reshape(rho, (1, -1))
    # "abstract" albedo should be in [0,1], so just normalise
    rho /= rho.max()

    # (re)create 2D fields
    rho2D[inside] = rho
    plt.imshow(rho2D, cmap='Greys_r')
    plt.axis('off')
    plt.title('Albedo')
    plt.show()

    n1 = np.zeros(mask.shape)
    n2 = np.zeros(mask.shape)
    n3 = np.ones(mask.shape)

    n1[inside] = M[0]
    n2[inside] = M[1]
    n3[inside] = M[2]

    # if I want to smooth the normal field, one of the calls (but not both)
    # would do the job.
    # n1, n2, n3 = ps_utils.smooth_normal_field(n1, n2, n3, mask, tau=0.1, iters=10, verbose=True)
    # n1, n2, n3 = ps_utils.tichonov_regularisation_normal_field(n1, n2, n3, 0.1, mask, tau=0.01, iters=10)

    f, (axn1, axn2, axn3) = plt.subplots(1, 3)
    axn1.imshow(n1)
    axn1.set_title('y-component of normal field.')
    axn2.imshow(n2)
    axn2.set_title('x-component of normal field.')
    axn3.imshow(n3)
    axn3.set_title('z-component of normal field.')
    for ax in axn1, axn2, axn3:
        ax.axis('off')
    plt.show()

    z = ps_utils.unbiased_integrate(n1, n2, n3, mask)
    # If I want to use the alternate integration method (Fourier based)
    # z = ps_utils.simchony_integrate(n1, n2, n3, mask)
    # simple dirty trick to solve convex / concave ambiguity (Yvain). 
    # Sometimes works
    if z[0, 0] > z[z.shape[0] // 2, z.shape[1] // 2]:
        z = -z
    ps_utils.display_surface(z, rho2D)
    # if Mayavi does not work: comment the line above
    # uncomment line below (slow!)
    # ps_utils.display_surface_matplotlib(z)

    return M, rho, z


def run_all():
    for dataset in all_datasets:
        I, mask, S, thresh = load_dataset(dataset)
        display_images_and_light(I, S)
        standard_reconstruction(I, mask, S)
        reconstruction_with_ransac(I, mask, S, thresh)


def run_Buddha():
    """ RANSAC produces problematic results with Buddha?"""
    I, mask, S, thresh = load_dataset('Buddha')
    M, rho, z = standard_reconstruction(I, mask, S)
    M2, rho2, z = reconstruction_with_ransac(I, mask, S, thresh)

def run_vase2():
    I, mask, S, thresh = load_dataset('shiny_vase2')
    M, rho, z = standard_reconstruction(I, mask, S)
    M2, rho2, z = reconstruction_with_ransac(I, mask, S, thresh)


if __name__ == "__main__":
    run_all()