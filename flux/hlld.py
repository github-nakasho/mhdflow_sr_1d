#!/usr/bin/env/ python3

import numpy as np

from common import Const
from convert import Convert

class HLLD:
    def __init__(self, ixmax):
        self.gm = Const.GAMMA
        self.gmr = self.gm / (self.gm-1)
        self.eps = Const.EPS
        self.conv = Convert(ixmax)
        self.max_iter = 20
        self.speed_limit = 1.0e-6

    def compute_left_right_state(self, Vlr):
        # set primitive variables @ right-face -----
        vn = Vlr[1]
        vt = Vlr[2]
        vu = Vlr[3]
        pg = Vlr[4]
        bnc = Vlr[5]
        bt = Vlr[6]
        bu = Vlr[7]
        # ----- set primitive variables @ right-face
        # compute v^2, B^2, v dot b, Lorents factor
        v2 = vn ** 2 + vt ** 2 + vu ** 2
        b2 = bnc ** 2 + bt ** 2 + bu ** 2 
        vb = vn * bnc + vt * bt + vu * bu
        # compute total pressure
        pt = pg + 0.5 * (b2*(1-v2)+vb**2)
        # compute Ul, Ur
        Ulr = self.conv.VtoU(Vlr)
        # compute Fl, Fr
        Flr = self.conv.UVtoF(Ulr, Vlr, b2, vb, v2)
        return Ulr, Flr, pt, bnc
        
    def compute_fs_mode(self, Vlr):
        # compute fast/slow mode signal speeds from Leismann 2005-----
        v2 = Vlr[1] * Vlr[1] + Vlr[2] * Vlr[2] + Vlr[3] * Vlr[3]
        b2 = Vlr[5] * Vlr[5] + Vlr[6] * Vlr[6] + Vlr[7] * Vlr[7]
        vb = Vlr[1] * Vlr[5] + Vlr[2] * Vlr[6] + Vlr[3] * Vlr[7]
        p = Vlr[4]
        ro = Vlr[0]
        vx = Vlr[1]
        h = 1 + self.gmr * p / ro
        bb2 = b2 * v2 + vb * vb
        hast = h + bb2 / ro
        cs2 = self.gm * p / (ro*h)
        ca2 = bb2 / (ro*hast)
        r = cs2 * vb * vb / (ro*hast) * (1-v2)
        om2 = cs2 + ca2 - cs2 * ca2
        den = 1 - v2 * om2 - r
        num1 = r - om2 * (1-v2)
        num2 = (1-om2) * vx * vx - den
        # compute lambda_fms eq. (27)
        vfp = vx * (1-om2) / den + np.sqrt(num1*num2) / den
        vfm = vx * (1-om2) / den - np.sqrt(num1*num2) / den
        # ----- compute fast/slow mode signal speeds from Leismann 2005
        return vfp, vfm

    def compute_middle_state(self, slr, Ulr, Flr, ptc, bnc): 
        # compute R components eq. (17) - (20)
        r = slr * Ulr - Flr
        # compute coefficients eq. (26) - (30)
        a = r[1] - slr * r[4] + ptc * (1-slr**2)
        g = r[6] ** 2 + r[7] ** 2
        c = r[2] * r[6] + r[3] * r[7]
        q = - a - g + bnc ** 2 * (1-slr**2)
        x = bnc * (a*slr*bnc+c) - (a+g) * (slr*ptc+r[4])
        # compute velocities eq. (23) - (25)
        vn = (bnc*(a*bnc+slr*c)-(a+g)*(ptc+r[1])) / x
        vt = (q*r[2]+r[6]*(c+bnc*(slr*r[1]-r[4]))) / x
        vu = (q*r[3]+r[7]*(c+bnc*(slr*r[1]-r[4]))) / x
        # compute B-fields eq. (21)
        bt = (r[6]-bnc*vt) / (slr-vn)
        bu = (r[7]-bnc*vu) / (slr-vn)
        # compute w eq. (31)
        w = ptc + (r[4]-(vn*r[1]+vt*r[2]+vu*r[3])) / (slr-vn)
        # if np.any(w<0):
        #     print("w is negarive")
        #     print((r[4]-(vn*r[1]+vt*r[2]+vu*r[3])) / (slr-vn))
        #     print(vn*r[1]+vt*r[2]+vu*r[3])
        #     print(slr-vn)
        #     import sys
        #     sys.exit()
        # compute v dot B
        vb = vn * bnc + vt * bt + vu * bu
        # compute conserved variables eq. (32) - (34)
        d = r[0] / (slr-vn)
        e = (r[4]+ptc*vn-vb*bnc) / (slr-vn)
        mn = (e+ptc) * vn - vb * bnc
        mt = (e+ptc) * vt - vb * bt
        mu = (e+ptc) * vu - vb * bu
        # compute Um 
        Um = np.zeros(Ulr.shape)
        Um[0] = d
        Um[1] = mn
        Um[2] = mt
        Um[3] = mu
        Um[4] = e
        Um[5] = bnc
        Um[6] = bt
        Um[7] = bu
        return vn, vt, vu, bt, bu, w, d, e, mn, mt, mu, Um, r
    
    def compute_central_state(self, dm, vnm, vtm, vum, mnm, ptc, em, bnc, btm, bum, btc, buc, knc, ktc, kuc, etac):
        # compute v eq. (47)
        kc2 = knc ** 2 + ktc ** 2 + kuc ** 2
        kbc = knc * bnc + ktc * btc + kuc * buc
        tmp = (1-kc2) / (etac-kbc)
        vnc = knc - bnc * tmp
        vtc = ktc - btc * tmp
        vuc = kuc - buc * tmp
        # compute D eq. (50)
        dc = dm * (knc-vnm) / (knc-vnc)
        # compute E eq. (51)
        vbc = vnc * bnc + vtc * btc + vuc * buc
        ec = (knc*em-mnm+ptc*vnc-vbc*bnc) / (knc-vnc)
        # compute m vector eq. (52)
        mnc = (ec+ptc) * vnc - vbc * bnc
        mtc = (ec+ptc) * vtc - vbc * btc
        muc = (ec+ptc) * vuc - vbc * buc
        # compute Uc
        Uc = np.zeros((8, len(dm)))
        Uc[0] = dc
        Uc[1] = mnc
        Uc[2] = mtc
        Uc[3] = muc
        Uc[4] = ec
        Uc[5] = bnc
        Uc[6] = btc
        Uc[7] = buc
        return vnc, vtc, vuc, dc, ec, mnc, mtc, muc, Uc
    
    def find_pressure(self, sl, sr, Ul, Ur, Fl, Fr, ptl, ptr, ptc, bnc):
        # compute middle right (mr) state -----
        vnmr, vtmr, vumr, \
        btmr, bumr, \
        wmr, dmr, emr, \
        mnmr, mtmr, mumr, \
        Umr, Rr = self.compute_middle_state(sr, Ur, Fr, ptc, bnc)
        # compute sigma vector eq. (35), (42)
        etacr = np.sign(bnc) * np.sqrt(wmr)
        # ----- compute middle right (mr) state
        # compute middle left (ml) state -----
        vnml, vtml, vuml, \
        btml, buml, \
        wml, dml, eml, \
        mnml, mtml, muml, \
        Uml, Rl = self.compute_middle_state(sl, Ul, Fl, ptc, bnc)
        # compute sigma vector eq. (35), (41)
        etacl = - np.sign(bnc) * np.sqrt(wml)
        # ----- compute middle left (ml) state
        # compute Kcr vector eq. (41)
        kncr = (Rr[1]+ptc+Rr[5]*etacr) / (sr*ptc+Rr[4]+bnc*etacr)
        ktcr = (Rr[2]+Rr[6]*etacr) / (sr*ptc+Rr[4]+bnc*etacr)
        kucr = (Rr[3]+Rr[7]*etacr) / (sr*ptc+Rr[4]+bnc*etacr)
        # Kx = sa (speed of Alfven wave)
        sar = kncr
        # compute Kcl vector eq. (41)
        kncl = (Rl[1]+ptc+Rl[5]*etacl) / (sl*ptc+Rl[4]+bnc*etacl)
        ktcl = (Rl[2]+Rl[6]*etacl) / (sl*ptc+Rl[4]+bnc*etacl)
        kucl = (Rl[3]+Rl[7]*etacl) / (sl*ptc+Rl[4]+bnc*etacl)
        # Kx = sa (speed of Alfven wave)
        sal = kncl
        # compute B-fields @ central region eq. (45)
        dk = kncr - kncl + self.eps
        btc = ((btmr*(kncr-vnmr)+bnc*vtmr)-(btml*(kncl-vnml)+bnc*vtml)) / dk
        buc = ((bumr*(kncr-vnmr)+bnc*vumr)-(buml*(kncl-vnml)+bnc*vuml)) / dk
        # compute center right (cr) state
        vncr, vtcr, vucr, \
        dcr, ecr, \
        mncr, mtcr, mucr, \
        Ucr = self.compute_central_state(dmr, 
                                                vnmr, vtmr, vumr, 
                                                mnmr, ptc, emr, 
                                                bnc, btmr, bumr, btc, buc, 
                                                kncr, ktcr, kucr, 
                                                etacr)
        # compute center left (cl) state
        vncl, vtcl, vucl, \
        dcl, ecl, \
        mncl, mtcl, mucl, \
        Ucl = self.compute_central_state(dml, 
                                                vnml, vtml, vuml, 
                                                mnml, ptc, eml, 
                                                bnc, btml, buml, btc, buc, 
                                                kncl, ktcl, kucl, 
                                                etacl)
        # get entropy wave speed (= average of left-side speed and right-side speed) 
        sm = 0.5 * (vncl+vncr)
        # judge unphysical solution or NOT
        failed = self.judgement_hlld(sl, sr, sal, sar, vnml, vnmr, vncl, vncr, wml, wmr, ptc)
        return vncl - vncr, sal, sar, btc, buc, sm, failed

    def initial_guess(self, ptl, ptr, bnc, sl, sr, Ul, Ur, Fl, Fr):
        ptc0 = np.zeros(ptl.shape)
        ptmax = np.maximum(ptl, ptr)
        # set true & false array
        ta = bnc ** 2 / ptmax < 0.01
        fa = ta == False | np.isnan(bnc**2/ptmax)
        # initial guess for Bx -> 0 limit
        a = sr - sl
        rr = sr * Ur - Fr
        rl = sl * Ul - Fl
        b = rr[4] - rl[4] + sr * rl[1] - sl * rr[1]
        c = rl[1] * rr[4] - rr[1] * rl[4]
        tmp = b * b - 4 * a * c
        tmp = np.maximum(tmp, 0)
        ptc0[ta] = 0.5 * (-b[ta]+np.sqrt(tmp[ta])) / a[ta]
        # compute conservative variables of HLL state
        Uhll = (sr*Ur-sl*Ul+Fl-Fr) / (sr-sl)
        # get primitive variables of HLL state
        Vhll = self.conv.UtoV(Uhll)
        # compute total pressure
        b2 = Vhll[5] * Vhll[5] + Vhll[6] * Vhll[6] + Vhll[7] * Vhll[7]
        v2 = Vhll[1] * Vhll[1] + Vhll[2] * Vhll[2] + Vhll[3] * Vhll[3]
        vb = Vhll[1] * Vhll[5] + Vhll[2] * Vhll[6] + Vhll[3] * Vhll[7]
        ptc0[fa] = Vhll[4][fa] + 0.5 * (b2[fa]*(1-v2[fa])+vb[fa]**2)
        return ptc0
    
    def judgement_hlld(self, sl, sr, sal, sar, vml, vmr, vcl, vcr, wl, wr, p):
        success = vcl - sal > - self.speed_limit
        success &= sal - vcr > - self.speed_limit
        success &= sl - vml < 0
        success &= sr - vmr > 0
        success &= sal - sl > -self.speed_limit
        success &= sr - sar > -self.speed_limit
        success &= wr - p > 0
        success &= wl - p > 0
        failed = success == False
        return failed

    def make_flux(self, Vl, Vr, ix, order):
        F = np.zeros(Vl.shape)
        eps = Const.EPS
        # get right state
        Ur, Fr, ptr, bnc = self.compute_left_right_state(Vr)
        # get left state
        Ul, Fl, ptl, bnc = self.compute_left_right_state(Vl)
        # compute left & right fast mode wave speeds
        vfpr, vfmr = self.compute_fs_mode(Vr)
        vfpl, vfml = self.compute_fs_mode(Vl)
        # compute propagation speeds of Riemann fan
        sl = np.minimum(0, vfml, vfmr)
        sr = np.maximum(0, vfpl, vfpr)
        # set true arrays
        tl = sl >= 0
        tr = sr <= 0
        tm = tl | tr
        tm = tm == False
        for m in range(8):
            F[m][tl] = Fl[m][tl]
            F[m][tr] = Fr[m][tr]
        if np.any(tm==True):
            # set switch variable of hll
            switch_to_hll = np.zeros(Vl[0].shape)
            # initial guess of total pressure
            ptc0 = self.initial_guess(ptl, ptr, bnc, sl, sr, Ul, Ur, Fl, Fr)
            # find total pressure -----
            ptc = ptc0
            f0, sal, sar, btc, buc, sm, failed = self.find_pressure(sl, sr, Ul, Ur, Fl, Fr, ptl, ptr, ptc0, bnc)
            # set true array
            ta = (abs(f0) > eps) & (switch_to_hll == False)
            fa = ta == False
            ptc[ta] = 1.025 * ptc0[ta]
            dp = np.zeros(len(bnc))
            for i in range(self.max_iter):
                f, sal, sar, btc, buc, sm, failed = self.find_pressure(sl, sr, Ul, Ur, Fl, Fr, ptl, ptr, ptc, bnc)
                # if iteration is too many, switch to HLL
                dp[ta] = (ptc[ta]-ptc0[ta]) / (f[ta]-f0[ta]) * f[ta]
                ptc0[ta] = ptc[ta]
                f0[ta] = f[ta]
                ptc[ta] -= dp[ta]
                # set negative pressure or nan array
                ne = ptc < 0 | np.isnan(ptc)
                ptc[ne] = eps
                if np.any(abs(dp[ta])<1.0e-6*ptc[ta]) or np.any(abs(f[ta])<1.0e-6):
                    break
            # set switch
            switch_to_hll = failed
            ptc[fa] = ptc0[fa]
            # set true arrays
            tml = sal >= - self.speed_limit
            tmr = sar <= self.speed_limit
            tc = tml | tmr
            tc = tc == False
            if np.any(tml==True):
                # get ml state
                vnl, vtl, vul, \
                btl, bul, \
                wl, dl, el, \
                mnl, mtl, mul, \
                Uml, Rl = self.compute_middle_state(sl, Ul, Fl, ptc, bnc)
                Fml = Fl + sl * (Uml - Ul)
                for m in range(8):
                    F[m][tml] = Fml[m][tml]
            if np.any(tmr==True):
                # get mr state
                vnr, vtr, vur, \
                btr, bur, \
                wr, dr, er, \
                mnr, mtr, mur, \
                Umr, Rr = self.compute_middle_state(sr, Ur, Fr, ptc, bnc)
                Fmr = Fr + sr * (Umr - Ur)                    
                for m in range(8):
                    F[m][tmr] = Fmr[m][tmr]
            if np.any(tc==True):
                # set true array 
                tcl = sm > 0
                tcr = tcl == False
                if np.any(tcl==True):
                    # get ml state
                    vnl, vtl, vul, \
                    btl, bul, \
                    wl, dl, el, \
                    mnl, mtl, mul, \
                    Uml, Rl = self.compute_middle_state(sl, Ul, Fl, ptc, bnc)
                    # compute sigma vector eq. (35), (42)
                    etal = - np.sign(bnc) * np.sqrt(wl)
                    # compute Kcl vector eq. (41)
                    knl = (Rl[1]+ptc+Rl[5]*etal) / (sl*ptc+Rl[4]+bnc*etal)
                    ktl = (Rl[2]+Rl[6]*etal) / (sl*ptc+Rl[4]+bnc*etal)
                    kul = (Rl[3]+Rl[7]*etal) / (sl*ptc+Rl[4]+bnc*etal)
                    # get mr state
                    vnr, vtr, vur, \
                    btr, bur, \
                    wr, dr, er, \
                    mnr, mtr, mur, \
                    Umr, Rr = self.compute_middle_state(sr, Ur, Fr, ptc, bnc)
                    # compute sigma vector eq. (35), (42)
                    etar = np.sign(bnc) * np.sqrt(wr)
                    # compute Kcr vector eq. (41)
                    knr = (Rr[1]+ptc+Rr[5]*etar) / (sr*ptc+Rr[4]+bnc*etar)
                    ktr = (Rr[2]+Rr[6]*etar) / (sr*ptc+Rr[4]+bnc*etar)
                    kur = (Rr[3]+Rr[7]*etar) / (sr*ptc+Rr[4]+bnc*etar)
                    # compute B-fields @ central region eq. (45)
                    dk = knr - knl + self.eps
                    btc = ((btr*(knr-vnr)+bnc*vtr)-(btl*(knl-vnl)+bnc*vtl)) / dk
                    buc = ((bur*(knr-vnr)+bnc*vur)-(bul*(knl-vnl)+bnc*vul)) / dk
                    # get cl state
                    vn, vt, vu, \
                    d, e, \
                    mn, mt, mu, \
                    Uc = self.compute_central_state(dl, vnl, vtl, vul, \
                                                        mnl, ptc, el, \
                                                        bnc, btl, bul, btc, buc, \
                                                        knl, ktl, kul, etal)
                    # convert state to flux
                    Fcl = Fl + sl * (Uml-Ul) + sal * (Uc-Uml)
                    for m in range(8):
                        F[m][tcl] = Fcl[m][tcl]
                if np.any(tcr==True):
                    # get ml state
                    vnl, vtl, vul, \
                    btl, bul, \
                    wl, dl, el, \
                    mnl, mtl, mul, \
                    Uml, Rl = self.compute_middle_state(sl, Ul, Fl, ptc, bnc)
                    # compute sigma vector eq. (35), (41)
                    etal = - np.sign(bnc) * np.sqrt(wl)
                    # compute Kcl vector eq. (41)
                    knl = (Rl[1]+ptc+Rl[5]*etal) / (sl*ptc+Rl[4]+bnc*etal)
                    ktl = (Rl[2]+Rl[6]*etal) / (sl*ptc+Rl[4]+bnc*etal)
                    kul = (Rl[3]+Rl[7]*etal) / (sl*ptc+Rl[4]+bnc*etal)
                    # get mr state
                    vnr, vtr, vur, \
                    btr, bur, \
                    wr, dr, er, \
                    mnr, mtr, mur, \
                    Umr, Rr = self.compute_middle_state(sr, Ur, Fr, ptc, bnc)
                    # compute sigma vector eq. (35), (42)
                    etar = np.sign(bnc) * np.sqrt(wr)
                    # compute Kcr vector eq. (41)
                    knr = (Rr[1]+ptc+Rr[5]*etar) / (sr*ptc+Rr[4]+bnc*etar)
                    ktr = (Rr[2]+Rr[6]*etar) / (sr*ptc+Rr[4]+bnc*etar)
                    kur = (Rr[3]+Rr[7]*etar) / (sr*ptc+Rr[4]+bnc*etar)
                    # compute B-fields @ central region eq. (45)
                    dk = knr - knl + self.eps
                    btc = ((btr*(knr-vnr)+bnc*vtr)-(btl*(knl-vnl)+bnc*vtl)) / dk
                    buc = ((bur*(knr-vnr)+bnc*vur)-(bul*(knl-vnl)+bnc*vul)) / dk
                    # get cr state
                    vn, vt, vu, \
                    d, e, \
                    mn, mt, mu, \
                    Uc = self.compute_central_state(dr, vnr, vtr, vur, \
                                                        mnr, ptc, er, \
                                                        bnc, btr, bur, btc, buc, \
                                                        knr, ktr, kur, etar)
                    # convert state to flux
                    Fcr = Fr + sr * (Umr-Ur) + sar * (Uc-Umr)
                    for m in range(8):
                        F[m][tcr] = Fcr[m][tcr]
        # if switch_to_hll == True, set HLL flux
        for m in range(8):
            F[m][switch_to_hll] = (sr[switch_to_hll]*Fl[m][switch_to_hll]
                                    -sl[switch_to_hll]*Fr[m][switch_to_hll]
                                    +sr[switch_to_hll]*sl[switch_to_hll]
                                    *(Ur[m][switch_to_hll]-Ul[m][switch_to_hll])) \
                                    / (sr[switch_to_hll]-sl[switch_to_hll])
        return F
