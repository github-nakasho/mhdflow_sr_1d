#!/usr/bin/env/ python3

import numpy as np

from scipy.optimize import root

from common import Const


class Recovery:
    def __init__(self, ixmax):
        self.wstart = [10] * ixmax
        self.wsave = np.array([10] * ixmax)
        
    def lo(self, b2, m2, s, w):
        wb = w + b2
        num = s * s * (wb+w) + m2 * w * w
        den = wb * wb * w * w
        tmp = num / den
        tmp[np.where(tmp>1-0.00001)] = 1 - 0.00001
        lo = 1 / np.sqrt(1-tmp)
        return lo

    def dlodw(self, b2, m2, s, w):
        wb = w + b2
        lo = self.lo(b2, m2, s, w)
        num = 2 * s * s * (3*w*wb+b2*b2) + m2 * w ** 3
        den = 2 * w ** 3 * wb ** 3
        return - lo ** 3 * num / den

    def pg(self, b2, d, m2, s, w):
        gmr = Const.GAMMA / (Const.GAMMA-1)
        lo = self.lo(b2, m2, s, w)
        return (w-d*lo) / (gmr*lo*lo)

    def dpgdw(self, b2, d, m2, s, w):
        gmr = Const.GAMMA / (Const.GAMMA-1)
        lo = self.lo(b2, m2, s, w)
        dlodw = self.dlodw(b2, m2, s, w)
        return (lo*(1+d*dlodw)-2*w*dlodw) / (gmr*lo**3)

    def dfwdw(self, b2, d, m2, s, w):
        dpgdw = self.dpgdw(b2, d, m2, s, w)
        lo = self.lo(b2, m2, s, w)
        dlodw = self.dlodw(b2, m2, s, w)
        return 1 - dpgdw + b2 / lo ** 3 * dlodw + s * s / w ** 3

    def fw(self, b2, d, e, m2, s, w):
        pg = self.pg(b2, d, m2, s, w)
        lo = self.lo(b2, m2, s, w)
        return w - pg + (1-0.5/lo**2) * b2 - s * s / (2*w*w) - e

    def w_ini(self, b2, e, m2, w):
        wb = 2 * w + b2
        return m2 - w * w + wb * (wb-2*e)

    def get_w(self, b2, d, e, m2, s):
        w_ini = lambda x: self.w_ini(b2, e, m2, x)
        self.wstart = [10] * len(b2)
        w = np.array(self.wstart)
        # w_ini = root(w_ini, [10]*len(b2), method='broyden1')
        # w = w_ini.x + d
        for _ in range(1000):
            dw = self.fw(b2, d, e, m2, s, w) / self.dfwdw(b2, d, m2, s, w)
            w_post = w - dw
            if np.all(dw<Const.EPS):
                break
            w = abs(w_post)
            self.wsave = w
        w = abs(w_post)
        return w