from PIL import Image
from numpy import *
from scipy.ndimage import filters
from pylab import *

# Read image
im = array(Image.open('../assignments/assignment1/Img001_diffuse_smallgray.png'))

imres = filters.gaussian_filter(im, 5)

# Show result image as gray scale image
gray()
imshow(imres)
show()


