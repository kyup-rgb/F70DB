# -*- coding: utf-8 -*-
"""
Created on Wed May 19 20:40:43 2021

@author: SavvasM
"""
import numpy as np

def Grad_Image(x):
    
    x_temp = x[1:, :] - x[0:-1,:]
    dux = np.c_[x_temp.T,np.zeros(np.size(x_temp,1))]
    dux = dux.T
    x_temp = x[:,1:] - x[:,0:-1]
    duy = np.c_[x_temp,np.zeros((np.size(x_temp,0),1))]
    return  np.concatenate((dux,duy),axis=0)
