#!/usr/bin/env/ python3

import numpy as np

from convert import Convert

def Check(V):
    conv = Convert(len(V[0]))
    V[4] = np.maximum(V[4], 1e-4)
    for _ in range(20):
        v2 = V[1] ** 2 + V[2] ** 2 + V[3] ** 2
        V[1][v2>1] *= 0.1
        V[2][v2>1] *= 0.1
        V[3][v2>1] *= 0.1
        v2 = V[1] ** 2 + V[2] ** 2 + V[3] ** 2
        if np.all(v2<1):
            break 
    U = conv.VtoU(V)
    return U, V
