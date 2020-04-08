#!/usr/bin/env/ python3

import numpy as np

from convert import Convert


class InitialCondition:
    def __init__(self, x, ix, order):
        self.x = x
        self.ixmax = ix + 2 * (order-1)
        self.V = np.zeros((8, self.ixmax))
        self.conv = Convert(self.ixmax)

    def B(self):
        V = self.V
        ixmax = self.ixmax
        for i in range(int(ixmax/2)):
            V[0][i] = 1.0
            V[1][i] = 0.0
            V[2][i] = 0.0
            V[3][i] = 0.0
            V[4][i] = 1.0
            V[5][i] = 0.5
            V[6][i] = 1.0
            V[7][i] = 0.0
        for i in range(int(ixmax/2), ixmax):
            V[0][i] = 0.125
            V[1][i] = 0.0
            V[2][i] = 0.0
            V[3][i] = 0.0
            V[4][i] = 0.1
            V[5][i] = 0.5
            V[6][i] = -1.0
            V[7][i] = 0.0
        U = self.conv.VtoU(V)
        return U, V

    def Sod(self):
        V = self.V
        ixmax = self.ixmax
        for i in range(int(ixmax/2)):
            V[0][i] = 10.0
            V[1][i] = 0.0
            V[2][i] = 0.0
            V[3][i] = 0.0
            V[4][i] = 40 / 3
            V[5][i] = 0.0
            V[6][i] = 0.0
            V[7][i] = 0.0
        for i in range(int(ixmax/2), ixmax):
            V[0][i] = 1.0
            V[1][i] = 0.0
            V[2][i] = 0.0
            V[3][i] = 0.0
            V[4][i] = 2 / 3 * 1e-6
            V[5][i] = 0.0
            V[6][i] = 0.0
            V[7][i] = 0.0
        U = self.conv.VtoU(V)
        return U, V

    def Test(self):
        V = self.V
        ixmax = self.ixmax
        for i in range(int(ixmax/2)):
            V[0][i] = 1.0
            V[1][i] = 0.5
            V[2][i] = 0.0
            V[3][i] = 0.0
            V[4][i] = 1.0
            V[5][i] = 0.0
            V[6][i] = 0.0
            V[7][i] = 0.0
        for i in range(int(ixmax/2), ixmax):
            V[0][i] = 1.0
            V[1][i] = 0.0
            V[2][i] = 0.0
            V[3][i] = 0.0
            V[4][i] = 1.0
            V[5][i] = 0.0
            V[6][i] = 0.0
            V[7][i] = 0.0
        U = self.conv.VtoU(V)
        return U, V
