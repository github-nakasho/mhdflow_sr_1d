#!/usr/bin/env/ python3

import numpy as np

from common import Const
from recovery import Recovery


class Convert:
    def __init__(self, ixmax):
        self.rec = Recovery(ixmax)

    def PtoF(self, ro, vx, vy, vz, bx, by, bz, pt, en):
        F = np.zeros((8, len(ro)))
        F[0] = ro * vx
        F[1] = ro * vx * vx + pt - bx * bx
        F[2] = ro * vy * vx - by * bx
        F[3] = ro * vz * vx - bz * bx
        F[4] = (en+pt) * vx - (vy*by+vz*bz) * bx
        F[5] = 0
        F[6] = by * vx - vy * bx
        F[7] = bz * vx - vz * bx
        return F

    def UtoV(self, U):
        gmr = Const.GAMMA / (Const.GAMMA-1)
        V = np.zeros(U.shape)
        b2 = U[5] * U[5] + U[6] * U[6] + U[7] * U[7]
        d = U[0]
        e = U[4]
        m2 = U[1] * U[1] + U[2] * U[2] + U[3] * U[3]
        s = U[1] * U[5] + U[2] * U[6] + U[3] * U[7]
        w = self.rec.get_w(b2, d, e, m2, s)
        lo = self.rec.lo(b2, m2, s, w)
        wb = w + b2
        V[0] = d / lo
        V[1] = 1 / wb * (U[1]+s/w*U[5])
        V[2] = 1 / wb * (U[2]+s/w*U[6])
        V[3] = 1 / wb * (U[3]+s/w*U[7])
        V[4] = (w-d*lo) / (gmr*lo*lo)
        V[5] = U[5]
        V[6] = U[6]
        V[7] = U[7]
        return V

    def UVtoF(self, U, V, b2, vb, v2):
        F = np.zeros(U.shape)
        lo = 1 / np.sqrt(1-v2)
        bx = V[5] / lo + lo * vb * V[1]
        by = V[6] / lo + lo * vb * V[2]
        bz = V[7] / lo + lo * vb * V[3]
        bb2 = b2 / (lo*lo) + vb * vb
        pt = V[4] + 0.5 * bb2
        F[0] = U[0] * V[1]
        F[1] = U[1] * V[1] - V[5] * bx / lo + pt
        F[2] = U[2] * V[1] - V[5] * by / lo
        F[3] = U[3] * V[1] - V[5] * bz / lo
        F[4] = U[1]
        F[5] = 0.0
        F[6] = U[6] * V[1] - V[2] * U[5]
        F[7] = U[7] * V[1] - V[3] * U[5]
        return F

    def VtoU(self, V):
        U = np.zeros(V.shape)    
        gmr = Const.GAMMA / (Const.GAMMA-1)
        v2 = V[1] * V[1] + V[2] * V[2] + V[3] * V[3]
        b2 = V[5] * V[5] + V[6] * V[6] + V[7] * V[7]
        vb = V[1] * V[5] + V[2] * V[6] + V[3] * V[7]
        lo = 1 / np.sqrt(1-v2)
        dhlo2 = V[0] * (1+gmr*V[4]/V[0]) * lo * lo
        U[0] = lo * V[0]
        U[1] = (dhlo2+b2) * V[1] - vb * V[5]
        U[2] = (dhlo2+b2) * V[2] - vb * V[6]
        U[3] = (dhlo2+b2) * V[3] - vb * V[7]
        U[4] = dhlo2 - V[4] + 0.5 * b2 + 0.5 * (v2*b2-vb*vb)
        U[5] = V[5]
        U[6] = V[6]
        U[7] = V[7]
        return U
