###
# This program is a simple implementation of the KLT point tracking algorithm.
# Not all aspects of the algorithm is implemented and there are some assumptions 
# which could be removed by improving the code.
# The program runs through a small image sequence found in data/ (borrowed 
# from Pao-Lung Tsai). 
#
#  By Kim Steenstrup Pedersen, 2013
###

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import filters
from scipy.ndimage.interpolation import map_coordinates
import copy


def gradient(img, sigma):
    """Computes the gradient image at scale sigma."""
    Ix = filters.gaussian_filter(img, sigma, (0, 1))
    Iy = filters.gaussian_filter(img, sigma, (1, 0))    
    return Ix, Iy


def smallestEigenvalue(G):
    """Compute the smallest eigenvalue and return."""
    e, v = np.linalg.eigh(G.reshape((2,2)))
    return min(e)
    
def checkFeature(G, threshold):
    """Compute the smallest eigenvalue and check if bigger than threshold.
       Returns a boolean.
    """
    e, v = np.linalg.eigh(G)
    return min(e) > threshold

        
def findextrema(im, thres_max):
    """Finds local extrema using 4-neighbors and check that it is bigger than thres_max."""
    n, m = im.shape
    extrema_list = []
    for r in range(1, n-1):
        for c in range(1, m-1):
            # Check that (r,c) is a maxima in 4-neighbors
            if im[r,c] > im[r-1, c] and im[r,c] > im[r+1, c] and im[r,c] > im[r, c-1] and im[r,c] > im[r, c+1]:
                if im[r,c] > thres_max:
                    extrema_list.append( (r, c) )
    return extrema_list

        
def nonoverlap(features, sigma, window_size):
    """Do nothing. TODO: Add code to remove overlapping features. 
       Potentially sort after feature strength and remove weaker points first."""
    return features


def goodFeaturesToTrack(img, sigma = 1, window_size = 15, threshold = 50):
    """Detects good features to track at scale sigma and feature window_size 
       All features are non-overlapping and the window is centered on the feature
       location.
       
       Window function is a box filter.
       
       Returns center position of detected features.
    """
 
    Ix, Iy = gradient(img, sigma)
    Ix2 = Ix**2
    Iy2 = Iy**2
    IxIy = Ix * Iy

    g1 = filters.uniform_filter(Ix2, window_size, output=None, mode='constant', cval=0.0)
    g2 = filters.uniform_filter(IxIy, window_size, output=None, mode='constant', cval=0.0)
    g3 = filters.uniform_filter(Iy2, window_size, output=None, mode='constant', cval=0.0)    
    G = np.dstack((g1, g2, g2, g3))
    
    B = filters.generic_filter(G, smallestEigenvalue, (1,1,4), mode='constant', cval=1.0)
        
    features = findextrema(B[:,:,2], threshold)
    features = nonoverlap(features, sigma, window_size)
    
    print "Number of feature candidates found: ", len(features)
    
    # This is the one!
#    plt.figure()
#    plt.gray()
#    plt.imshow(B[:,:,2])
#    row, col =zip(*features)
#    plt.plot(col, row, 'r.')
    
    #print np.max(B[:,:,2])
    #print np.min(B[:,:,2])
    
    return features
    


def computeLinearSystem(I, J, sigma):
    """Compute the G matrix and e vector."""
    Jx, Jy = gradient(J, sigma)
    
    # Remove previously added border
    Jx = Jx[3*sigma:-3*sigma, 3*sigma:-3*sigma]
    Jy = Jy[3*sigma:-3*sigma, 3*sigma:-3*sigma]
    
    # compute G and e   
    Jx2 = Jx**2
    Jy2 = Jy**2
    JxJy = Jx * Jy
    G = np.array([[np.sum(Jx2), np.sum(JxJy)],[np.sum(JxJy), np.sum(Jy2)]], dtype=np.float64)
    e = np.array([[np.sum((I[3*sigma:-3*sigma, 3*sigma:-3*sigma]-J[3*sigma:-3*sigma, 3*sigma:-3*sigma])*Jx)], [np.sum((I[3*sigma:-3*sigma, 3*sigma:-3*sigma]-J[3*sigma:-3*sigma, 3*sigma:-3*sigma])*Jy)]], dtype=np.float64)        
    return G, e
    

def klttracker(I, J, featuresI, sigma=1, window_size=15, threshold = 50):
    """This function is the main step in the KLT tracker algorithm. 
       featuresI are a list of feature coordinates in image I that we want 
       to track in image J. Assumes window_size is an odd integer and sigma
       is an integer.
       
       Returns a list of tracked features in image J.
    """
    
    converge_eps = 50000
    featuresJ = []
    halfwidth = window_size / 2 # Assuming window_size is an odd integer
    Jdim = J.shape
    
    imgCoordr, imgCoordc = np.mgrid[0:Jdim[0], 0:Jdim[1]]
    
    # Loop over features
    for (row, col) in featuresI:
        #print row, col
        
        # Check feature + border is inside image otherwise throw out
        if row < (halfwidth + 3*sigma) or row > (Jdim[0] - halfwidth - 3*sigma):
            continue
        if col < (halfwidth + 3*sigma) or col > (Jdim[1] - halfwidth - 3*sigma):
            continue
                    
        # Use griddata to extract interpolated image patch to non integer coordinates
        grid_r, grid_c = np.mgrid[(row-halfwidth-3*sigma):(row+halfwidth+3*sigma):((window_size+6*sigma)*1j), (col-halfwidth-3*sigma):(col+halfwidth+3*sigma):((window_size+6*sigma)*1j)]        
        coordsI = np.vstack( ( grid_r.reshape(1, (window_size+6*sigma)**2 ) , grid_c.reshape(1, (window_size+6*sigma)**2 ) ) )
        coordsJ = coordsI
        patchI = map_coordinates(I, coordsI, order=1, prefilter=False).reshape(window_size+6*sigma, window_size+6*sigma)
        patchJ = map_coordinates(J, coordsJ, order=1, prefilter=False).reshape(window_size+6*sigma, window_size+6*sigma)
        
        new_row = row
        new_col = col
        isconverged = False
        iter = 1
        while not isconverged:        
            # Compute G matrix on patches from I and J
            G, e = computeLinearSystem(patchI, patchJ, sigma)
    
            # Perform test on minimum eigenvalue
            if not checkFeature(G, threshold):
                #continue # Skip this feature if threshold test fails
                break
            
            # Compute displacement and store new coordinate in featuresJ
            d = np.dot(np.linalg.inv(G), e)        
            new_row = new_row+d[1]
            new_col = new_col+d[0]
            
            # Check for convergence
            if iter >= 5:
                isconverged = True # But drop feature
                # print "iter >= 5 - dropping feature"
            else:
                grid_r, grid_c = np.mgrid[(new_row-halfwidth-3*sigma):(new_row+halfwidth+3*sigma):((window_size+6*sigma)*1j), (new_col-halfwidth-3*sigma):(new_col+halfwidth+3*sigma):((window_size+6*sigma)*1j)]        
                coordsJ = np.vstack( ( grid_r.reshape(1, (window_size+6*sigma)**2 ) , grid_c.reshape(1, (window_size+6*sigma)**2 ) ) )
                patchJ = map_coordinates(J, coordsJ, order=1, prefilter=False).reshape(window_size+6*sigma, window_size+6*sigma)
            
                diff = np.sum( (patchJ - patchI)**2 )
                #print diff
                if diff > converge_eps: # Not converged yet - repeat
                    iter += 1
                else: # Have converged - save point and stop iterating
                    featuresJ.append( (new_row, new_col) )
                    isconverged = True
        
    print "Number of feature candidates found in J: ", len(featuresJ)
    return featuresJ
    



# Main script
I = np.array(Image.open('data/img0.pgm'), dtype=np.float64)

plt.ion() # Interactive mode - draw plots immediately

# Key parameters of the KLT tracker
sigma = 1 
window_size = 15 
threshold = 250

# Find features to track in first frame
featuresI = goodFeaturesToTrack(I, sigma, window_size, threshold)

# Show images
plt.figure()
plt.gray()
plt.imshow(I)
if len(featuresI) > 0:
    row, col = zip(*featuresI)
    plt.plot(col, row, 'r.')
    plt.draw()


# Loop over images
for no in range(1,10):
    print 'Processing image ' + 'data/img' + str(no) + '.pgm'
    J = np.array(Image.open('data/img' + str(no) + '.pgm'), dtype=np.float64)
    featuresJ = klttracker(I, J, featuresI, sigma, window_size, threshold)

    plt.figure()
    plt.gray()
    plt.imshow(J)
    if len(featuresJ) > 0:
        row, col = zip(*featuresJ)
        plt.plot(col, row, 'r.')
        plt.draw()

    # Override I with J and load new image
    del I
    I = J.copy()
    featuresI = copy.deepcopy(featuresJ)



plt.figure()
plt.gray()
plt.imshow(J)
if len(featuresJ) > 0:
    row, col = zip(*featuresJ)
    plt.plot(col, row, 'r.')
    plt.draw()

# Show all figures
#plt.show()

raw_input('Press a key')

plt.ioff()