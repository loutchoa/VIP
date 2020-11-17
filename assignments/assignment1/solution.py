#import PIL
from PIL import Image
from numpy import *
from scipy.ndimage import filters
from pylab import *


def laplacian(im, sigma):
    """Computes the Laplacian image at scale sigma."""
    imxx = filters.gaussian_filter(im, sigma, (0, 2))
    imyy = filters.gaussian_filter(im, sigma, (2, 0))
    lap = sigma**2 * (imxx + imyy)
    return lap


def findextrema(im, thres_min, thres_max):
    """Finds local extrema using 4-neighbors."""
    n,m = im.shape
    extrema_list = []
    for r in range(1, n-1):
        for c in range(1, m-1):
            if im[r,c] > im[r-1, c] and im[r,c] > im[r+1, c] and im[r,c] > im[r, c-1] and im[r,c] > im[r, c+1]:
                if im[r,c] > thres_max:
                    extrema_list.append( (r, c, im[r,c]) )
            elif im[r,c] < im[r-1, c] and im[r,c] < im[r+1, c] and im[r,c] < im[r, c-1] and im[r,c] < im[r, c+1]:
                if im[r,c] < thres_min:
                    extrema_list.append( (r, c, im[r,c]) )
    return extrema_list
    

def detectblobs(im, sigma, thres_min, thres_max):
    """Detects Laplacian blobs at fixed scale sigma and discard points using thresholds."""
    lap = laplacian(im, sigma)
    extrema_list = findextrema(lap, thres_min, thres_max)
    return extrema_list


def extract_patch(im, coord, patch_size):
    """Extract a patch at (row, col) in im of size patch_size (a 2-tuple).
    
    Assumes that border cases have been removed and that patch_size is odd integer.
    """
    r, c =coord
    pnr, pnc = patch_size
    F = im[(r-pnr/2):(r+pnr/2 + 1), (c-pnc/2):(c+pnc/2 + 1)]
    return F

def NCC(F1, F2):
    """Normalized cross correlation. 
    
    F1 and F2 must be of equal size."""
    n, m = F1.shape
    res = 1 - (F1.flatten()-F1.mean()).dot( F2.flatten()-F2.mean() ) / (F1.std() * F2.std() * n * m)
    #res = 1 - (F1.flatten()-F1.mean()).dot( F2.flatten()-F2.mean() ) / (F1.std() * F2.std())
    return res
    

# Slow as hell!
def find_match(im1, extrema_list1, im2, extrema_list2, patch_size):
    """Find matching points using raw patch descriptor and normalized cross correlation"""
    pnr, pnc = patch_size
    im1n, im1m = im1.shape
    im2n, im2m = im2.shape
    res = []
    
    for coord1 in extrema_list1:
        row1, col1, lap1 = coord1
        if ( (row1 - pnr) > 0 and (col1 - pnc) > 0 ) and ( (im1n-1-row1) > pnr and (im1m-1-col1) > pnc ):
            F1 = extract_patch(im1, (row1, col1), patch_size)
            best = (0, 0, 0, 0, 1000) # rbest1, cbest1, rbest2, cbest2, dbest
            secondBest = (0, 0, 0, 0, 999)
            for coord2 in extrema_list2:
                row2, col2, lap2 = coord2
                #print "im1: ", row1, col1
                #print "im2: ", row2, col2
                if ( (row2 - pnr) > 0 and (col2 - pnc) > 0 )  and ( (im2n-1-row2) > pnr and (im2m-1-col2) > pnc ):
                    F2 = extract_patch(im2, (row2, col2), patch_size)
                    dist = NCC(F1, F2)
                    if dist < best[4]:
                        secondBest = best
                        best = (row1, col1, row2, col2, dist)
                #else:             
                #    print "Im2 point at boundary thrown away"
            if (best[4] / secondBest[4]) < 0.8: # Then a match
                res.append(best)
        #else:
        #    print "Im1 point at boundary thrown away"
            
    return res
    


# Parameters of script
sigma = 2
thres_max = 30
thres_min = -30
patch_size = (9, 9) # Row, Col


# Read image
im1 = array(Image.open('Img001_diffuse_smallgray.png'), dtype=float)
extrema_list1 = detectblobs(im1, sigma, thres_min, thres_max)

im2 = array(Image.open('Img002_diffuse_smallgray.png'), dtype=float)
extrema_list2 = detectblobs(im2, sigma, thres_min, thres_max)


# Show result image as gray scale image

imres1 = filters.gaussian_filter(im1, sigma)
figure()
gray()
imshow(imres1)

# Plot extrema points
y, x, lap = zip(*extrema_list1)
plot(x, y, 'r.')


imres2 = filters.gaussian_filter(im2, sigma)
figure()
gray()
imshow(imres2)

# Plot extrema points
y, x, lap = zip(*extrema_list2)
plot(x, y, 'r.')



# Match extrema
match_list = find_match(im1, extrema_list1, im2, extrema_list2, patch_size)
y1, x1, y2, x2, d = zip(*match_list)

# Visualize the matches
imcomb=append(im1, im2, axis=1)

figure()
gray()
imshow(imcomb)
origoy, origox = im1.shape
plot(x1, y1, 'r.')
plot(array(x2) + origox, y2, 'b.')

for elem in match_list:
    y1, x1, y2, x2, d = elem
    plot([x1, array(x2) + origox], [y1, y2], 'y-')


# Show all figures
show()



