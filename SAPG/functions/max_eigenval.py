# -*- coding: utf-8 -*-
"""
Created on Mon May 24 19:40:47 2021

@author: SavvasM
"""
import numpy as np
from scipy.linalg import norm

def max_eigenval(A, At, im_size, tol, max_iter, verbose):
#computes the maximum eigen value of the compund operator AtA
    x = np.random.randn(im_size,im_size)
    x = x/norm(np.ravel(x),2)
    init_val = 1
    
    for k in range(0,max_iter):
        y = A(x)
        x = At(y)
        val = norm(np.ravel(x),2)
        rel_var = np.abs(val-init_val)/init_val
        if (verbose > 1):
            print('Iter = {}, norm = {}',k,val)
        
        if (rel_var < tol):
           break
        
        init_val = val
        x = x/val
    
    if (verbose > 0):
        print('Norm = {}', val)
    
    return val