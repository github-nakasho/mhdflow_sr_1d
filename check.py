#!/usr/bin/env/ python3

import numpy as np

from convert import Convert

def Check(V):
    conv = Convert(len(V[0]))
    V[4] = np.maximum(V[4], 1e-4)
    U = conv.VtoU(V)
    return U, V
