#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Project:

File:
    
Description:
    
"""

import numpy as np
import matplotlib.pyplot as plt

__author__ = "Francois Lauze, University of Copenhagen"  
__date__ = "Sun Dec  1 23:13:34 2019"
__version = "0.0.0"


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Project:

File:
    
Description:
    
"""

import numpy as np
from matplotlib import pyplot as plt
import random
from skimage.measure import LineModelND, ransac

__author__ = "Francois Lauze, University of Copenhagen"  
__date__ = "Sun Dec  1 22:44:55 2019"
__version = "0.0.0"




np.random.seed(seed=1)

# generate coordinates of line
x = np.arange(-200, 200)
y = 0.2 * x + 20
data = np.column_stack([x, y])

# add gaussian noise to coordinates
noise = np.random.normal(size=data.shape)
data += 0.5 * noise
data[::2] += 5 * noise[::2]
data[::4] += 20 * noise[::4]

print(data.shape)
# add faulty data
faulty = np.array(200 * [(180., -100)])
faulty += 100 * np.random.normal(size=faulty.shape)
data[:faulty.shape[0]] = faulty

# fit line using all data
model = LineModelND()
model.estimate(data)

# robustly fit line only using inlier data with RANSAC algorithm
model_robust, inliers = ransac(data, LineModelND, min_samples=2,
                               residual_threshold=1, max_trials=1000)
outliers = inliers == False
#
## generate coordinates of estimated models
line_x = np.arange(-150, 250)
line_y = model.predict_y(line_x)
line_y_robust = model_robust.predict_y(line_x)

fig, ax = plt.subplots()
ax.plot(data[inliers, 0], data[inliers, 1], '.b', alpha=0.6,
        label='Inlier data', markersize=9)
ax.plot(data[outliers, 0], data[outliers, 1], '.r', alpha=0.6,
        label='Outlier data')
#ax.plot(line_x, line_y, '-k', label='Line model from all data')
#ax.plot(line_x, line_y_robust, '-b', label='Robust line model')

idx = random.sample(range(len(data)),2)
i1, i2 = idx
ax.plot(data[idx,0], data[idx,1], 'k',   linewidth=6.0, label='Randomly sampled')
ax.plot(data[idx,0], data[idx,1], '.k', markersize=25)

threshold=1.0
# compute inliers for currently selected model using standard least square, not TLS
X = data[:,0]
Y = data[:,1]
#A = np.array([[(X**2).sum(), X.sum()],[X.sum(), len(X)]])
#B = np.array([(X*Y).sum(), X.sum()])
#a,b = np.linalg.inv(A)@B
## draw lines y=ax + b
#x_1 = X.min()
#x_2 = X.max()
#y_1 = a*x_1/3.0 + b
#y_2 = a*x_2/3.0 + b
#ax.plot([x_1, x_2], [y_1, y_2], 'g', linewidth=2.0, label='from naive least-squares')

x1 = X[i1]
x2 = X[i2]
y1 = Y[i1]
y2 = Y[i2]
print("coordinates:", x1,x2,y1,y2)
a = (y2 - y1)/(x2 - x1)
b = (y1*x2 - y2*x1)/(x2 - x1)
print("Coefficients", a, b)
print("type(x1)", type(x1))
x0 = x1 - (x2-x1)/4
x3 = x2 + (x2-x1)/4
y0 = a*x0 + b
y3 = a*x3 + b
ax.plot([x0, x3], [y0, y3], 'g', linewidth=2.0, label='from naive least-squares')
r = (Y - a*X - b)**2

idx = np.where(r < threshold)
print(idx)
ax.plot(X[idx], Y[idx], '.y', markersize=16, label='model inliers')

ax.legend(loc='lower left', fontsize=20)
ax.axis('off')
plt.show()
























