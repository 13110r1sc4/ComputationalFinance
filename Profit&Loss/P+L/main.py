#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 15:56:30 2024

@author: leonardorisca
"""

"""

' *************************************************************************** '
' ***************************   PROFIT & LOSS ******************************* '
' *************************************************************************** '
' ******* Vega Gamma, Heston ************************************************ '
' *************************************************************************** '

BREAKDOWN PSEUDOCODE

1) calculate option price with XX (Heston & VG)

2) compute implied volatility using option prices from point 1 and BSM model

3) the IV frm point 2 is used as the volatility parameter in BSM model for 
   underlying trajectories, which we want to study at a point in time of 
   6 months from now: 
       
       for each of the trajectories compute the option price and consequently
       the profit/loss

4) Build the histogram for the RV = P/L aand compute VaR (19%, 5%, 1%)
    

PYHTON FILES STRUCTURE

The main file will output the histogram and the three var values
    
"""

import pandas as pd
import sys
import numpy as np
from CFLib.timer import Timer
from math import *
from scipy.stats import norm
from CFLib.config import get_input_parms
from sys import stdout as cout
from H_VG import do_mc, ControlVariates, AS
import matplotlib.pyplot as plt
from CFLib.stats import stats
from CFLib.euro_opt import euro_put
import seaborn as sns

# ------------------------------------

def cn_put( T, sigma, kT):
    s    = sigma*sqrt(T)
    d    = ( np.log(kT) + .5*s*s)/s
    return norm.cdf(d)

def an_put( T, sigma, kT):
    s    = sigma*sqrt(T)
    d    = ( np.log(kT) + .5*s*s)/s
    return norm.cdf(d-s)

def FwEuroPut(T, vSigma, vKt):
    return ( vKt* cn_put( T, vSigma, vKt) - an_put( T, vSigma, vKt) )

def FwEuroCall(T, sigma, vkT):
    return FwEuroPut(T, sigma, vkT) + 1. - vkT

def impVolFromMCPut(vPrice, T, vKt):

    scalar = isinstance(vKt, float)
    if scalar: vKt = np.array([vKt])

    vSl = np.zeros(vKt.shape[0])
    vPl = np.maximum(vKt - 1., 0.0) # call

    vSh = np.ones(vKt.shape[0])
    while True:
        vPh = FwEuroPut(T, vSh, vKt)
        if ( vPh > vPrice).all(): break
        vSh = 2*vSh

    # d = vSh-vSl
    # d/2^N < eps
    # d < eps* 2^N
    # N > log(d/eps)/log(2)
    eps = 1.e-08
    d   = vSh[0]-vSl[0]
    N   = 2+int(log(d/eps)/log(2))

    for n in range(N):
        vSm  = .5*(vSh + vSl)
        vPm  = FwEuroPut(T, vSm, vKt)
        mask = vPm > vPrice # if the new price is higher than the price i want,
                            # i will lower the high value for vol
        vSh[mask] = vSm[mask]
        vSl[~mask] = vSm[~mask] # ~ is a not?

    
    if scalar: return .5*(vSh + vSl)[0]
    return .5*(vSh + vSl)
# ------------------------------------

def BS_trj(Obj, nt, So, T, J, sigma):

    DT = T/nt
    S  = np.ndarray(shape = ( nt+1, J), dtype=np.double)

    X  = Obj.normal( -.5*sigma*sigma*DT, sigma*sqrt(DT), (nt,J))

    # for j in range(0, J): 
    #   S[0,j] = So
    S[0] = So

    for n in range(nt):
        # for j in range(0, J): 
        #   S[n+1,j] = S[n,j] * exp( X[n,j] )
        S[n+1] = S[n] * np.exp( X[n] )

    return S
# ------------------------------------

def run_PL( argv ):
    
    parms = get_input_parms(sys.argv)

    if "help" in parms:
      usage()
      return
    
    name   = parms.get("out", None)
    if name is None: fp = cout
    else:            fp = open(name,"w")

    '****************************************************************************'
    '****************************************************************************'
    
    models =["H", "VG"]
        
    '****************** ALL *******************'
    So     =  float(parms.get("s", "1.0"))
    k      =  float(parms.get("k", "1.03"))
    sigma  =  float(parms.get("v", ".3"))
    T      =  float(parms.get("T", "1.13"))
    r      =  float(parms.get("r", "0.01"))
    q      =  float(parms.get("q", "0.0"))
    seed   =  int(parms.get("seed", "1"))
    nt     =  int(parms.get("nt", "20"))

    
    '****************** VG *******************'
    eta_vg    =  float(parms.get("eta_vg", "0.1494"))
    nu_vg     =  float(parms.get("nu_vg", "0.0626"))
    theta_vg  =  float(parms.get("theta_vg", "-0.6635"))
    n         =  int(parms.get("n", "22"))

    
    '*************** HESTON *******************'
    lmbda  =  float(parms.get("lmbda", "7.7648"))
    nubar  =  float(parms.get("nubar", "0.0601"))
    eta    =  float(parms.get("eta", "2.0170"))
    nu_o   =  float(parms.get("nu_o", "0.0475"))
    rho    =  float(parms.get("rho", -0.6952))
    nv     =  int(parms.get("nv", "14"))
    ns     =  int(parms.get("ns", "10"))
    dt     =  float(parms.get("dt", 1./365))
    
    '****************************************************************************'
    '****************************************************************************'
    
    fp.write("@ %-12s %8.4f\n" %("So", So))
    fp.write("@ %-12s %8.4f\n" %("r" , r))
    fp.write("@ %-12s %8.4f\n" %("q" , q))
    fp.write("\n");
    fp.write("@ %-12s %8.4f\n" %("lmbda" , lmbda))
    fp.write("@ %-12s %8.4f\n" %("nubar" , nubar))
    fp.write("@ %-12s %8.4f\n" %("eta" , eta))
    fp.write("@ %-12s %8.4f\n" %("nu_o" , nu_o))
    fp.write("@ %-12s %8.4f\n" %("rho" , rho))
    fp.write("@ %-12s %8.4f\n" %("dt" , dt))
    fp.write("\n");
    fp.write("@ %-12s %8.4f\n" %("n" , n))
    fp.write("\n");
    fp.write("@ %-12s %8.4f\n" %("nv" , nv))
    fp.write("@ %-12s %8.4f\n" %("ns" , ns))
    fp.write("@ %-12s %8.4f\n" %("nt" , nt))
    fp.write("\n");
    
    N  = 1 << n
    NS = 1 << ns
    NV = 1 << nv
    
    mcPut_mkt    = {x: []            for x in models}
    asPut_mkt    = {x: []            for x in models}
    mcPut_bsm    = {x: []            for x in models}
    asPut_bsm    = {x: []            for x in models}
    mcIV         = {x: []            for x in models}
    asIV         = {x: []            for x in models}
    mc_PL        = {x: np.zeros(N)   for x in models}
    as_PL        = {x: np.zeros(N)   for x in models}


    T1 = Timer()
    T1.start()
    
    rand = np.random.RandomState()
    rand.seed(seed)
    
    Fw = So*exp((r-q)*T)
    kT = k/Fw

    for model in models:
                        
        mod          =  str(parms.get("mod", model))
        
        mcPut_mkt[mod] = do_mc(So=So, T=T, r = r, q = q, sigma=sigma, strike=k, N=N, fp=fp, seed=seed, nt=nt, lmbda=lmbda, nubar=nubar, eta=eta, nu_o=nu_o, rho=rho, dt=dt, NV=NV, NS=NS, mod=model, eta_vg=eta_vg, nu_vg=nu_vg, theta_vg=theta_vg)
        mcIV[mod] = impVolFromMCPut(mcPut_mkt[mod], T, kT)
        asPut_mkt[mod] = AS(sigma, r, q, So, k, T, lmbda, nubar, eta, nu_o, rho, mod, eta_vg, nu_vg, theta_vg)
        asIV[mod] = impVolFromMCPut(asPut_mkt[mod], T, kT)

        nt =  int(parms.get("nt", "10"))
        dt =  float(parms.get("dt", T/nt))
        tn = np.zeros(shape=(nt+1, 1))
        for i in range(len(tn)):
            tn[i] = round(i*dt, 3)
        
        sigma  =  float(parms.get("v", mcIV[mod]))
        S_mc = BS_trj(rand, nt, So, T, N, sigma)        
        mcBS_as = euro_put(So, r, q, T, sigma, k)
 
        sigma  =  float(parms.get("v", asIV[mod]))
        S_as = BS_trj(rand, nt, So, T, N, sigma)
        asBS_as = euro_put(So, r, q, T, sigma, k)

    
        mc_vPayoff    = S_mc[-1] - kT
        as_vPayoff    = S_as[-1] - kT
        
        mcPut_bsm[mod] = np.where( -mc_vPayoff > 0.0, -mc_vPayoff, 0.0) * exp(-r * T) * Fw
        asPut_bsm[mod] = np.where( -as_vPayoff > 0.0, -as_vPayoff, 0.0) * exp(-r * T) * Fw
        mcPut_bsm[mod] = ControlVariates(S_mc[-1], mcPut_bsm[mod])
        asPut_bsm[mod] = ControlVariates(S_as[-1], asPut_bsm[mod])
        mc, stdc = stats(mcPut_bsm[mod])
        mcPut_bsm[mod] = mc
        mc, stdc = stats(asPut_bsm[mod])
        asPut_bsm[mod] = mc

        T_6m = 0.5
        mc_6m = np.where( -mc_vPayoff > 0.0, -mc_vPayoff, 0.0) * exp(-r * ( T - T_6m)) * Fw
        as_6m = np.where( -as_vPayoff > 0.0, -as_vPayoff, 0.0) * exp(-r * ( T - T_6m)) * Fw

        mc_PL[mod] = mcPut_mkt[mod] - np.where( -mc_vPayoff > 0.0, -mc_vPayoff, 0.0) * exp(-r * T) * Fw
        as_PL[mod] = asPut_mkt[mod] - np.where( -as_vPayoff > 0.0, -as_vPayoff, 0.0) * exp(-r * T) * Fw
        
        mc_VaR_10 = np.percentile(mc_PL[mod], 10)
        mc_VaR_5 = np.percentile(mc_PL[mod], 5)
        mc_VaR_1 = np.percentile(mc_PL[mod], 1)
        as_VaR_10 = np.percentile(as_PL[mod], 10)
        as_VaR_5 = np.percentile(as_PL[mod], 5)
        as_VaR_1 = np.percentile(as_PL[mod], 1)
        
        profit_mc = sum(1 for x in mc_PL[mod] if x > 0)/len(mc_PL[mod])
        profit_as = sum(1 for x in as_PL[mod] if x > 0)/len(as_PL[mod])
        
        fp.write("\n")
        fp.write("\n############### %3s MARKET  ###############\n" %(model))
        fp.write("\n######          MC         AS   \n")
        fp.write("\n")
        fp.write("# P_mkt     %7.5f   %7.5f \n" %(mcPut_mkt[mod], asPut_mkt[mod]))
        fp.write("# IV        %7.5f   %7.5f \n" %(mcIV[mod], asIV[mod]) )
        fp.write("# P_bs_mc   %7.5f   %7.5f \n" %(mcPut_bsm[mod], asPut_bsm[mod]) )
        fp.write("# P_bs_as   %7.5f   %7.5f \n" %(mcBS_as, mcBS_as))
        fp.write("\n")
        fp.write("\n################ %3s P&L  #################\n" %(model))
        fp.write("\n######          MC         AS   \n")
        fp.write("\n")
        fp.write("# Mean       %7.5f   %7.5f \n" %(mc_PL[mod].mean(), as_PL[mod].mean()))
        fp.write("# Profit(%%)  %7.5f   %7.5f \n" %(profit_mc, profit_as) )
        fp.write("\n")
        fp.write("\n################ %3s VaR  #################\n" %(model))
        fp.write("\n######          MC         AS   \n")
        fp.write("\n")
        fp.write("# 0.10       %7.5f   %7.5f \n" %(mc_VaR_10, as_VaR_10))
        fp.write("# 0.05       %7.5f   %7.5f \n" %(mc_VaR_5, as_VaR_5))
        fp.write("# 0.01       %7.5f   %7.5f \n" %(mc_VaR_1, as_VaR_1))
        fp.write("\n")
        fp.write("#---\n")
        fp.write("\n")
        fp.write("\n")

        plt.ioff()        
        
        fig, axs = plt.subplots(2, 1, figsize=(10, 10))
        fig.suptitle(f'BS trajectories of the underlying asset', fontsize=16)
        axs[0].plot(tn, S_mc, antialiased=False)
        axs[0].set_title(fr'$\sigma = IV_{{{mod}[MC]}}$', fontsize=13)    
        axs[0].grid(True, which='both')
        axs[1].plot(tn, S_as, antialiased=False)
        axs[1].set_title(fr'$\sigma = IV_{{{mod}[AS]}}$', fontsize=13)        
        axs[1].grid(True, which='both')
        fig.text(0.95, 0.95, fr'$N = 2^{{{n}}}$', ha='right', va='top', fontsize=12)
        plt.tight_layout()
        plt.show()
        
        fig, axs = plt.subplots(2, 1, figsize=(10, 10))
        fig.suptitle(f'BS put price 6 months from now conditional on the trajectory', fontsize=16)
        sns.kdeplot(mc_6m, ax=axs[0], color='blue', label=fr'$\sigma = IV_{{{mod}[MC]}}$')
        axs[0].set_title(fr'$\sigma = IV_{{{mod}[MC]}}$', fontsize=13)        
        axs[0].set_xlim([mc_6m.min(), mc_6m.max()])
        sns.kdeplot(as_6m, ax=axs[1], color='blue', label=fr'$\sigma = IV_{{{mod}[MC]}}$')
        axs[1].set_title(fr'$\sigma = IV_{{{mod}[AS]}}$', fontsize=13)        
        axs[1].set_xlim([as_6m.min(), as_6m.max()])
        fig.text(0.95, 0.95, fr'$N = 2^{{{n}}}$', ha='right', va='top', fontsize=12)
        plt.tight_layout()
        plt.show()

                
        fig, axs = plt.subplots(2, 2, figsize=(10, 10))
        fig.suptitle(f'P&L', fontsize=16)
        
        sns.kdeplot(mc_PL[mod], ax=axs[0,0], color='blue', label=fr'$\sigma = IV_{{{mod}[MC]}}$')
        axs[0,0].set_title(fr'$\sigma = IV_{{{mod}[MC]}}$', fontsize=13)        
        axs[0,0].set_xlim([mc_PL[mod].min(), mc_PL[mod].max()])

        sns.kdeplot(as_PL[mod], ax=axs[0,1], color='blue', label=fr'$\sigma = IV_{{{mod}[MC]}}$')
        axs[0,1].set_title(fr'$\sigma = IV_{{{mod}[AS]}}$', fontsize=13)        
        axs[0,1].set_xlim([as_PL[mod].min(), as_PL[mod].max()])
        
        axs[1,0].hist(mc_PL[mod], bins=70, edgecolor='black')
        axs[1,0].axvline(mc_VaR_10, color='r', linestyle='--', linewidth=2, label='VaR at 10%')
        axs[1,0].axvline(mc_VaR_5, color='orange', linestyle='--', linewidth=2, label='VaR at 5%')
        axs[1,0].axvline(mc_VaR_1, color='y', linestyle='--', linewidth=2, label='VaR at 1%')
        axs[1,0].set_title(fr'$\sigma = IV_{{{mod}[MC]}}$', fontsize=13)        
        axs[1,0].set_xlim([mc_PL[mod].min(), mc_PL[mod].max()])
        axs[1,0].legend()
        
        axs[1,1].hist(as_PL[mod], bins=70, edgecolor='black')
        axs[1,1].axvline(as_VaR_10, color='r', linestyle='--', linewidth=2, label='VaR at 10%')
        axs[1,1].axvline(as_VaR_5, color='orange', linestyle='--', linewidth=2, label='VaR at 5%')
        axs[1,1].axvline(as_VaR_1, color='y', linestyle='--', linewidth=2, label='VaR at 1%')
        axs[1,1].set_title(fr'$\sigma = IV_{{{mod}[AS]}}$', fontsize=13)        
        axs[1,1].set_xlim([as_PL[mod].min(), as_PL[mod].max()])
        axs[1,1].legend()
        plt.tight_layout()
        plt.show()
        
if __name__ == "__main__":
    run_PL( sys.argv )


