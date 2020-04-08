#!/usr/bin/env/ python3

import numpy as np

from common import Const


def SetDt(V, minlength, cfl):
    gm = Const.GAMMA
    gmr = gm / (gm-1)
    p = V[4]
    ro = V[0]
    v2 = V[1] * V[1] + V[2] * V[2] + V[3] * V[3]
    b2 = V[5] * V[5] + V[6] * V[6] + V[7] * V[7]
    vb = V[1] * V[5] + V[2] * V[6] + V[3] * V[7]    
    h = 1 + gmr * V[4] / V[0]
    cs2 = gm * p / (ro*h)
    lo = 1 / np.sqrt(1-v2)
    num = b2 / (lo*lo) + vb*vb
    den = ro + gmr * p + num
    ca2 = num / den
    r = cs2 * vb * vb / (lo*lo*den)
    om2 = cs2 + ca2 - cs2 * ca2
    v = V[1]
    vfx = v * (1-om2) / (1-v2*om2-r) + np.sqrt(((v2-1)*om2+r)*((v2-v*v)*om2+v*v-1+r))/(1-v2*om2-r)
    v = V[2]
    vfy = v * (1-om2) / (1-v2*om2-r) + np.sqrt(((v2-1)*om2+r)*((v2-v*v)*om2+v*v-1+r))/(1-v2*om2-r)
    v = V[3]
    vfz = v * (1-om2) / (1-v2*om2-r) + np.sqrt(((v2-1)*om2+r)*((v2-v*v)*om2+v*v-1+r))/(1-v2*om2-r)
    vf = np.sqrt(vfx**2+vfy**2+vfz**2)
    oneocfldt = max(vf/minlength)
    dt = cfl/oneocfldt
    return dt
