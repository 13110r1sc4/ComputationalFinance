#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 15:36:19 2024

@author: leonardorisca
"""

' *************************************************************************** '
' ********** CEV, Displ Diff, Heston MC & Analytical options prices ********* '
' *************************************************************************** '

import sys
from CFLib.timer import Timer
from sys import stdout as cout
from time import time
from math import *
import numpy as np
from CFLib.config import get_input_parms
from CFLib.stats import stats
from scipy.stats import norm
from CFLib.Heston import Heston
from CFLib.heston_evol import __mc_heston__
from CFLib.CIR import CIR, QT_cir_evol
import pyfeng as pf
# ------------------------------------------------

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

def euro_put(So, r, q, T, sigma, k):
    kT   = exp((q-r)*T)*k/So
    return So*exp(-q*T) * FwEuroPut( T, sigma, kT)

def euro_call(So, r, q, T, sigma, k):
    kT   = exp((q-r)*T)*k/So
    # ( exp(-rT)*K* cn_put( T, vSigma, vKt) - So*exp(-qT)*an_put( T, vSigma, vKt) )
    return So*exp(-q*T) * FwEuroCall(T, sigma, kT)

def AS(sigma, r, q, So, k, T, b, delta, lmbda, nubar, eta, nu_o, rho, mod):
    
    Fw = So*exp((r-q)*T)
    kT = k/Fw

    if mod == "CEV":
        cev = pf.Cev(sigma=sigma, beta=b, intr=r, divr=q)
        call = cev.price(strike=k, spot=So, texp=T, cp=1)
        put = cev.price(strike=k, spot=So, texp=T, cp=-1)
    elif mod == "DD":
        Y = So + delta
        call = euro_call( Y, r, q, T, sigma, k)
        put  = euro_put ( Y, r, q, T, sigma, k) 
    elif mod == "H":
        Xc = 50
        H = Heston(lmbda=lmbda, nubar=nubar, eta=eta, nu_o=nu_o, rho=rho)
        put = H.HestonPut(So, k, T, Xc, r, q)
        call = put + exp(-r*T)*Fw*(So - kT)
        
    if __name__ == "__main__":
        return call, put
    else:
        return call
    
# ------------------------------------------------

def ControlVariates(S_T, disc_payoff):
    cov   = np.cov(S_T, disc_payoff)[1,0]
    var_s = np.var(S_T)
    c_opt = cov/var_s
    R = (disc_payoff - c_opt*(S_T - S_T.mean())) # use this to calculate opt price 
    
    return R

def DDiff_trj(Obj, nt, So, T, J, sigma, delta):

    DT = T/nt
    S  = np.ndarray(shape = ( nt+1, J), dtype=np.double)

    X  = Obj.normal( -.5*sigma*sigma*DT, sigma*sqrt(DT), (nt,J))

    # for j in range(0, J): 
    #   S[0,j] = So
    S[0] = So + delta

    for n in range(nt):
        # for j in range(0, J): 
        #   S[n+1,j] = S[n,j] * exp( X[n,j] )
        S[n+1] = S[n] * np.exp( X[n] )

    return S


def CEV_trj(Obj, nt, So, T, J, sigma, b):
    
    DT = T/nt
    S  = np.ndarray(shape = ( nt+1, J), dtype=np.double)
    eps = So/100
    # define eps

    S[0] = So

    for n in range(nt):
        # for j in range(0, J): 
        #   S[n+1,j] = S[n,j] * exp( X[n,j] )
        
        # change dimension of X from (nt,J) to (1,J)
        
        X  = Obj.normal( -.5*sigma*sigma*(S[n]*S[n])**(b-1)*DT, sigma*(S[n]**(b-1))*sqrt(DT), (1,J))
        S[n+1] = S[n] * np.exp( X )
        # filtra per S < eps
        S[n+1] = np.maximum(S[n+1],eps)

    return S
    

def heston_trj( rand, So, lmbda, eta, nubar, nu_o, rho, Yrs, dt , NV, NS, Nt):
    
    cir = CIR(kappa=lmbda, sigma=eta, theta=nubar, ro = nu_o)

    nCir      = int(Yrs/dt)
    dt        = float(Yrs/nCir)
    Dt        = Yrs/Nt
    blockSize = (2 << 8 )
    if NV > blockSize : 
        NB = NV//blockSize
    else:
        NB = 1
        blockSize = NV

    rem = NV - NB*blockSize 

    S   = np.zeros(shape=(NV, Nt+1, NS), dtype=np.float64) 

    for nb in range(NB):
        vol, Ivol = QT_cir_evol( rand, cir, nCir, dt, Nt, Dt, blockSize)
        Ivol = Ivol.transpose()
        vol = vol.transpose()
        for n in range(blockSize):
            s = __mc_heston__( rand, So, vol[n], Ivol[n], cir, rho, Dt, NS )
            S[n + nb*blockSize] = s
    if rem > 0:
        vol, Ivol = QT_cir_evol( rand, cir, nCir, dt, Nt, Dt, rem)
        Ivol = Ivol.transpose()
        vol = vol.transpose()
        for n in range(rem):
            s = __mc_heston__( rand, So, vol[n], Ivol[n], cir, rho, Dt, NS )
            S[NB*blockSize+rem] = s

    return S
# ------------------------------------------------

def euro_vanilla(rand, So, r, q, b, T, sigma, k, N, nt, delta, lmbda, nubar, eta, nu_o, rho, dt, NV, NS, mod):
    
    if mod == "DD":
        delta=delta
    else:
        delta=0.0
  
    Fw = So*exp((r-q)*T)
    kT = k/Fw

    if mod == "CEV":
        S  = CEV_trj(rand, nt, So, T, N, sigma, b)
        vPayoff    = S[-1]-kT
        call    = np.where(  vPayoff > 0.0,  vPayoff, 0.0) * exp(-r*T)*Fw
        put     = np.where( -vPayoff > 0.0, -vPayoff, 0.0) * exp(-r*T)*Fw
    elif mod == "DD":
        S  = DDiff_trj(rand, nt, So, T, N, sigma, delta)
        vPayoff    = S[-1]-kT
        call    = np.where(  vPayoff > 0.0,  vPayoff, 0.0) * exp(-r*T)*Fw
        put     = np.where( -vPayoff > 0.0, -vPayoff, 0.0) * exp(-r*T)*Fw
    elif mod == "H":
        S  = heston_trj( rand, So, lmbda, eta, nubar, nu_o, rho, T, dt , NV, NS, nt) 
        vPayoff = S[:,-1,:] - kT
        call   = np.add.reduce(np.where(  vPayoff > 0.0,  vPayoff, 0.0), 1)/NS * exp(-r*T)*Fw
        put    = np.add.reduce(np.where(  -vPayoff > 0.0,  -vPayoff, 0.0), 1)/NS * exp(-r*T)*Fw
        S = np.add.reduce(S[:,:,:], 2).transpose()/NS
        
    call = ControlVariates(S[-1], call)
    put  = ControlVariates(S[-1], put)

    # call_cp = put  + exp(-r*T)*Fw*(1. - kT)
    # put_cp  = call + exp(-r*T)*Fw*(kT - 1.)
    call_cp = put  + exp(-r*T)*Fw*(So+delta - kT)
    put_cp  = call + exp(-r*T)*Fw*(kT - (So+delta))

    return  call, put, call_cp, put_cp
# -----------------------------------------------------

def do_mc( **keywrds ):

    So    = keywrds["So"]
    T     = keywrds["T"]
    nt    = keywrds["nt"]
    r     = keywrds["r"]
    q     = keywrds["q"]
    b     = keywrds["b"]
    delta = keywrds["delta"]
    k     = keywrds["strike"]
    N     = keywrds["N"]
    fp    = keywrds["fp"]
    sigma = keywrds["sigma"]
    Seed  = keywrds["seed"]
    lmbda = keywrds["lmbda"]
    nubar = keywrds["nubar"]
    eta   = keywrds["eta"]
    nu_o  = keywrds["nu_o"]
    rho   = keywrds["rho"]
    dt    = keywrds["dt"]
    NV    = keywrds["NV"]
    NS    = keywrds["NS"]
    mod   = keywrds["mod"]
    
    # Initialise random number with a seed
    # So we can have reproducible results
    rand = np.random.RandomState()
    rand.seed(Seed)
    
    ' MC EURO VANILLA '

    mcCall, mcPut, mcCall_cp, mcPut_cp = euro_vanilla(rand, So, r, q, b, T, sigma, k, N, nt, delta, lmbda, nubar, eta, nu_o, rho, dt, NV, NS, mod)
    
    if __name__ == "__main__":
        return stats(mcCall), stats(mcPut), stats(mcCall_cp), stats(mcPut_cp)
    else:
        if k < So*exp((r-q)*T):
            mc, stdc = stats(mcCall_cp)
        elif k >= So*exp((r-q)*T):
            mc, stdc = stats(mcCall)
        
        return mc
# ----------------------------------------------------

def usage():
    print("Computes via MC Call and Put option prices for the CEV, DD, Heston models")
    print("Usage: $> ./C_D_H.py [options]")
    print("Options:")
    print("    %-24s: this output" %("--help"))
    print("    %-24s: initial value of the underlying, defaults to 1.0" %("-s So"))
    print("    %-24s: option strike, defaults to 1.0" %("-k strike"))
    print("    %-24s: option strike, defaults to .20" %("-v volatility"))
    print("    %-24s: option maturity, defaults to 1.0" %("-T maturity"))
    print("    %-24s: interest rate, defaults to 0.0" %("-r ir"))
    print("    %-24s: dividend yield, defaults to 0.0" %("-q qy"))
    print("    %-24s: log_2 of the number of iterations defaults to 20" %("-n nrTrj" ) )
    print("    %-24s: random seed, defaults to 1234567" %("-seed nr"))
    print("    %-24s: output file, defaults to stdout" %("-out output file"))
    print("    %-24s: number of ti, defaults to 10" %("-nt nt"))
    print("    %-24s: elasticity of volatility, defaults to 1.0" %("-b beta"))
    print("    %-24s: shift, defaults to 0.0" %("-delta delta"))
    print("    %-24s: shift, defaults to 7.7648" %("-lmbda lmbda"))
    print("    %-24s: shift, defaults to 0.0601" %("-nubar nubar"))
    print("    %-24s: shift, defaults to 2.0170" %("-eta eta"))
    print("    %-24s: shift, defaults to 0.0475" %("-nu_o nu_o"))
    print("    %-24s: shift, defaults to -0.6952" %("-rho rho"))
    print("    %-24s: shift, defaults to 1./365" %("-dt dt"))
    print("    %-24s: shift, defaults to 14" %("-nv nv"))
    print("    %-24s: shift, defaults to 8" %("-ns ns"))

# ----------------------------------------------------
    
def run( argv ):

    parms = get_input_parms(argv)

    if "help" in parms:
        usage()
        return
    
    '***********************************************************************'
    '***********************************************************************'
    
    models = ["CEV", "DD"]

    '****************** ALL *******************'
    So     =  float(parms.get("s", "1.0"))
    k      =  float(parms.get("k", "1.0"))
    sigma  =  float(parms.get("v", ".3"))
    T      =  float(parms.get("T", 1.))
    r      =  float(parms.get("r", "0.0"))
    q      =  float(parms.get("q", "0.0"))
    seed   =  int(parms.get("seed", "1"))
    
    '**************** CEV, DD *****************'
    n      =  int(parms.get("n", "20"))
    nt     =  int(parms.get("nt", "10"))
    
    '****************** CEV *******************'
    b      =  float(parms.get("b", "0.8")) # max 4 9
    
    '******************* DD *******************'
    delta  =  float(parms.get("delta", "0.1"))
    
    '*************** HESTON *******************'
    lmbda  =  float(parms.get("lmbda", "7.7648"))
    nubar  =  float(parms.get("nubar", "0.0601"))
    eta    =  float(parms.get("eta", "2.0170"))
    nu_o   =  float(parms.get("nu_o", "0.0475"))
    rho    =  float(parms.get("rho", -0.6952))
    dt     =  float(parms.get("dt", 1./365))
    nv     =  int(parms.get("nv", "16"))
    ns     =  int(parms.get("ns", "8"))
    
    '***********************************************************************'
    '***********************************************************************'

    N = 1 << n
    NS = 1 << ns
    NV = 1 << nv

    name   = parms.get("out", None)
    if name is None: fp = cout
    else:            fp = open(name,"w")

    fp.write("@ %-12s %8.4f\n" %("So", So))
    fp.write("@ %-12s %8.4f\n" %("Strike", k))
    fp.write("@ %-12s %8.4f\n" %("sigma", sigma))
    fp.write("@ %-12s %8.4f\n" %("T" , T))
    fp.write("@ %-12s %8.4f\n" %("r" , r))
    fp.write("@ %-12s %8.4f\n" %("q" , q))
    fp.write("@ %-12s %8.4f\n" %("nt" , nt))
    fp.write("\n")
    fp.write("@ %-12s %8.4f\n" %("b" , b))
    fp.write("@ %-12s %8.4f\n" %("delta" , delta))    
    fp.write("\n")
    fp.write("@ %-12s %8.4f\n" %("lambda" , lmbda))
    fp.write("@ %-12s %8.4f\n" %("nubar" , nubar))
    fp.write("@ %-12s %8.4f\n" %("eta" , eta))
    fp.write("@ %-12s %8.4f\n" %("nu_o" , nu_o))
    fp.write("@ %-12s %8.4f\n" %("rho" , rho))
    fp.write("@ %-12s %8.4f\n" %("NV" , NV))
    fp.write("@ %-12s %8.4f\n" %("NS" , NS))
    fp.write("@ %-12s %8.4f\n" %("nt" , nt))
    fp.write("@ %-12s %8.4f\n" %("dt" , dt))
    fp.write("\n");

    T1 = Timer()
    T1.start()
    
    mc = ["mcCall", "mcPut", "mcCall_cp", "mcPut_cp"]
    analyt = ["Call", "Put"]
    stats = ["m", "std"]
    
    MC_OPT = {x: {MC: {s: [] for s in stats} for MC in mc} for x in models}
    AS_OPT = {x: {a: [] for a in analyt} for x in models}
    
    for model in models:
        
        [MC_OPT[model]["mcCall"]["m"], MC_OPT[model]["mcCall"]["std"]], [MC_OPT[model]["mcPut"]["m"], MC_OPT[model]["mcPut"]["std"]], [MC_OPT[model]["mcCall_cp"]["m"], MC_OPT[model]["mcCall_cp"]["std"]], [MC_OPT[model]["mcPut_cp"]["m"], MC_OPT[model]["mcPut_cp"]["std"]] = do_mc(
            So=So, T=T, r = r, q = q, b = b, sigma=sigma, strike=k, N=N, fp=fp, seed=seed, nt=nt, delta=delta, lmbda=lmbda, nubar=nubar, eta=eta, nu_o=nu_o, rho=rho, dt=dt, NV=NV, NS=NS,mod=model)
        
        [AS_OPT[model]["Call"], AS_OPT[model]["Put"]] = AS(sigma, r, q, So, k, T, b, delta, lmbda, nubar, eta, nu_o, rho, mod=model)


    end = T1.stop()
    
    
    for model in models:
        
        if model == "CEV" or model == "DD":
            N_trj = N
        elif model == "H":
            N_trj = NV
        
        fp.write("\n###################### %3s ######################\n" %(model))
        fp.write("\n#-- (S-%5.2f)^+ = %8.4f\n" %(k, AS_OPT[model]["Call"]))
        fp.write("%9s    %10s %3s %12s -- %5s\n" %("N", "E[(S-k)^+]", "+/-", "MC-err", "Op"))
        Op     = ( abs(MC_OPT[model]["mcCall"]["m"]-AS_OPT[model]["Call"]) < 3.*MC_OPT[model]["mcCall"]["std"]/sqrt(N_trj) )
        fp.write("%9d    %10.4f %3s %12.2e -- %5s\n" %( N, MC_OPT[model]["mcCall"]["m"], "+/-", MC_OPT[model]["mcCall"]["std"]/sqrt(N_trj), Op))
        fp.write("\n#----\n")

        # Call via Call-put parity
        fp.write("\n#-- (S-%5.2f)^+ = %8.4f   Call-Put parity\n" %(k, AS_OPT[model]["Call"]))
        fp.write("%9s    %10s %3s %12s -- %5s\n" %("N", "E[(S-k)^+]", "+/-", "MC-err", "Op"))
        Op     = ( abs(MC_OPT[model]["mcCall_cp"]["m"]-AS_OPT[model]["Call"]) < 3.*MC_OPT[model]["mcCall_cp"]["std"]/sqrt(N_trj) )
        fp.write("%9d    %10.4f %3s %12.2e -- %5s\n" %( N, MC_OPT[model]["mcCall_cp"]["m"], "+/-", MC_OPT[model]["mcCall_cp"]["std"]/sqrt(N_trj), Op))
        fp.write("\n#----\n")

        # Standard put
        fp.write("\n#-- (%5.2f - S)^+ = %8.4f\n" %(k, AS_OPT[model]["Put"]))
        fp.write("%9s    %10s %3s %12s -- %5s\n" %("N", "E[(S-k)^+]", "+/-", "MC-err", "Op"))
        Op  = ( abs(MC_OPT[model]["mcPut"]["m"]-AS_OPT[model]["Put"]) < 3.*MC_OPT[model]["mcPut"]["std"]/sqrt(N_trj) )
        fp.write("%9d    %10.4f %3s %12.2e -- %5s\n" %( N, MC_OPT[model]["mcPut"]["m"], "+/-", MC_OPT[model]["mcPut"]["std"]/sqrt(N_trj), Op))
        fp.write("\n#----\n")

        # Put via Call-put parity
        fp.write("\n#-- (%5.2f - S)^+ = %8.4f   Call-Put parity\n" %(k, AS_OPT[model]["Put"]))
        fp.write("%9s    %10s %3s %12s -- %5s\n" %("N", "E[(S-k)^+]", "+/-", "MC-err", "Op"))
        Op  = ( abs(MC_OPT[model]["mcPut_cp"]["m"]-AS_OPT[model]["Put"]) < 3.*MC_OPT[model]["mcPut_cp"]["std"]/sqrt(N_trj) )
        fp.write("%9d    %10.4f %3s %12.2e -- %5s\n" %( N, MC_OPT[model]["mcPut_cp"]["m"], "+/-", MC_OPT[model]["mcPut_cp"]["std"]/sqrt(N_trj), Op))
        fp.write("\n#----\n")
        
    fp.write("Elapsed: %10.4f sec.\n" %( end));

    if not name is None: print("@ %-12s: output written to '%s'" %("Info", name))


if __name__ == "__main__":
    run(sys.argv)