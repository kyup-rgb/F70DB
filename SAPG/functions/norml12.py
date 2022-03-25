# -*- coding: utf-8 -*-
"""
Created on Thu May 13 10:45:30 2021

@author: SavvasM
"""

import numpy as np

def tv(Dx):
    
    Dx=Dx.ravel()
    N = len(Dx)
    Dux = Dx[:int(N/2)]
    Dvx = Dx[int(N/2):N]
    tv = np.sum(np.sqrt(Dux**2 + Dvx**2))
    
    return tv