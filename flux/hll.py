#!/usr/bin/env/ python3

import numpy as np

from common import Const
from convert import Convert


class HLL:
    def __init__(self, ixmax):
        self.convert = Convert(ixmax)

    def make_flux(self, Vl, Vr, ix, order):
        F = np.zeros(Vl.shape)
        gm = Const.GAMMA
        gmr = gm / (gm-1)
        # left side ----------
        v2 = Vl[1] * Vl[1] + Vl[2] * Vl[2] + Vl[3] * Vl[3]
        b2 = Vl[5] * Vl[5] + Vl[6] * Vl[6] + Vl[7] * Vl[7]
        vb = Vl[1] * Vl[5] + Vl[2] * Vl[6] + Vl[3] * Vl[7]
        Ul = self.convert.VtoU(Vl)
        Fl = self.convert.UVtoF(Ul, Vl, b2, vb, v2)
        p = Vl[4]
        ro = Vl[0]
        vx = Vl[1]
        h = 1 + gmr * p / ro
        lo = 1 / np.sqrt(1-v2)
        bb2 = b2 / (lo*lo) + vb * vb
        hast = h + bb2 / ro
        cs2 = gm * p / (ro*h)
        ca2 = bb2 / (ro*hast)
        r = cs2 * vb * vb / (ro*hast*lo*lo)
        om2 = cs2 + ca2 - cs2 * ca2
        den = 1 - v2 * om2 - r
        num1 = r - om2 / (lo*lo)
        num2 = (1-om2) * vx * vx - den
        vflp = vx * (1-om2) / den + np.sqrt(num1*num2) / den
        vflm = vx * (1-om2) / den - np.sqrt(num1*num2) / den
        # ---------- left side
        # right side ----------
        v2 = Vr[1] * Vr[1] + Vr[2] * Vr[2] + Vr[3] * Vr[3]
        b2 = Vr[5] * Vr[5] + Vr[6] * Vr[6] + Vr[7] * Vr[7]
        vb = Vr[1] * Vr[5] + Vr[2] * Vr[6] + Vr[3] * Vr[7]
        Ur = self.convert.VtoU(Vr)
        Fr = self.convert.UVtoF(Ur, Vr, b2, vb, v2)
        p = Vr[4]
        ro = Vr[0]
        vx = Vr[1]
        h = 1 + gmr * p / ro
        lo = 1 / np.sqrt(1-v2)
        bb2 = b2 / (lo*lo) + vb * vb
        hast = h + bb2 / ro
        cs2 = gm * p / (ro*h)
        ca2 = bb2 / (ro*hast)
        r = cs2 * vb * vb / (ro*hast*lo*lo)
        om2 = cs2 + ca2 - cs2 * ca2
        den = 1 - v2 * om2 - r
        num1 = r - om2 / (lo*lo)
        num2 = (1-om2) * vx * vx - den
        vfrp = vx * (1-om2) / den + np.sqrt(num1*num2) / den
        vfrm = vx * (1-om2) / den - np.sqrt(num1*num2) / den
        # ----------- right side
        # propagation speed of Riemann fan ----------
        sl = np.minimum(0, vflm, vfrm)
        sr = np.maximum(0, vflp, vfrp)
        # ---------- propagation speed of Riemann fan
        # compute HLL flux
        for m in range(8):
            F[m] = (sr*Fl[m]-sl*Fr[m]+sr*sl*(Ur[m]-Ul[m])) / (sr-sl)
        return F
        