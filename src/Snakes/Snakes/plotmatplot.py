# -*- coding: utf-8 -*-
"""
plot to matrix to plot change of coordinates
@author: francois
"""

def plot2mat_point(x, y, nx):
    return nx-1-y, x
        
def mat2plot_point(i, j, nx):
    return j, nx-1-i
    