#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@project: VIP 2020 2021
    
@filename: calculations.py
    
@description: some calculatiosn for the assignment

@author: François Lauze, University of Copenhagen    
Created on Thu Nov  5 10:03:34 2020

"""

import numpy as np
import  matplotlib.pyplot as plt


__version__ = "0.0.1"
__author__ = "François Lauze"


P = np.eye(4)
f0, f1, f2, f3 = P

f0 = f0.copy()
f1 = f1.copy()
f2 = f2.copy()
f3 = f3.copy()

print(f1)

P[0,:] += f1 -2*f2
#P[1,:] += -f0 + f2 + 3*f3
#P[2,:] += 4*f0 + 2*f1 -6*f3
P[3,:] += -f0 -2*f1 -f2

print("\n",P)

print(np.linalg.det(P))
print()
A = np.array([[0,1,2,1],[0,0,1,-2],[0,0,0,2],[0,0,0,0]])
print("A = \n",A)
print("A^2 = \n",A@A)
print("A^3 = \n",A@A@A)
print("A^4 = \n",A@A@A@A)


print()
Q = P@A@(np.linalg.inv(P))
print(Q)

print()
print("Q = \n",Q)
print("Q^2 = \n",Q@Q)
print("Q^3 = \n",Q@Q@Q)
print("Q^4 = \n",Q@Q@Q@Q)

