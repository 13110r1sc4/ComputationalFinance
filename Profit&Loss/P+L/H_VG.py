#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 17:28:17 2024

@author: leonardorisca
"""

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
from CFLib.CIR import CIR, cir_evol
from CFLib.VG import VG
from CFLib.VG import vg_evol_step

# ------------------------------------------------

# def cn_put( T, sigma, kT):
#     s    = sigma*sqrt(T)
#     d    = ( np.log(kT) + .5*s*s)/s
#     return norm.cdf(d)

# def an_put( T, sigma, kT):
#     s    = sigma*sqrt(T)
#     d    = ( np.log(kT) + .5*s*s)/s
#     return norm.cdf(d-s)

# def FwEuroPut(T, vSigma, vKt):
#     return ( vKt* cn_put( T, vSigma, vKt) - an_put( T, vSigma, vKt) )

# def FwEuroCall(T, sigma, vkT):
#     return FwEuroPut(T, sigma, vkT) + 1. - vkT

# def euro_put(So, r, q, T, sigma, k):
#     kT   = exp((q-r)*T)*k/So
#     return So*exp(-q*T) * FwEuroPut( T, sigma, kT)

# def euro_call(So, r, q, T, sigma, k):
#     kT   = exp((q-r)*T)*k/So
#     # ( exp(-rT)*K* cn_put( T, vSigma, vKt) - So*exp(-qT)*an_put( T, vSigma, vKt) )
#     return So*exp(-q*T) * FwEuroCall(T, sigma, kT)

def AS(sigma, r, q, So, k, T, lmbda, nubar, eta, nu_o, rho, mod, eta_vg, nu_vg, theta_vg):
    
    Fw = So*exp((r-q)*T)
    kT = k/Fw

    if mod == "VG":   
        Xc = 50                                                        
        vg = VG(eta=eta_vg, nu=nu_vg, theta=theta_vg)
        put  = vg.VGPut(So, k, T, Xc, r, q)
        call = put + exp(-r*T)*Fw*(So - kT)
        
    elif mod == "H":
        Xc = 50
        H = Heston(lmbda=lmbda, nubar=nubar, eta=eta, nu_o=nu_o, rho=rho)
        put = H.HestonPut(So, k, T, Xc, r, q)
        call = put + exp(-r*T)*Fw*(So - kT)
        
    if __name__ == "__main__":
        return call, put
    else:
        return put
    
# ------------------------------------------------

def ControlVariates(S_T, disc_payoff):
    cov   = np.cov(S_T, disc_payoff)[1,0]
    var_s = np.var(S_T)
    c_opt = cov/var_s
    R = (disc_payoff - c_opt*(S_T - S_T.mean())) # use this to calculate opt price
    
    # print(np.corrcoef(S_T,disc_payoff)[0,1])
        
    return R
# ------------------------------------------------

def VG_trj( rand, So, vg, Nt, Dt, N ):

    S  = np.ndarray(shape = (Nt+1, N), dtype=np.double ) # S[N, L] in fortran matrix notation
    S[0] = So
    for n in range(Nt):
        S[n+1] = vg_evol_step(rand, S[n], vg, Dt, N)

    return S

def QT_cir_evol( rand, cir, L, dt, Nt, DT, N): # with condition to keep 

    s   = cir.sigma
    th  = cir.theta
    k   = cir.kappa
    ro  = cir.ro

    PSI_c    = 1.5
    V      = np.ndarray(shape = ( L+1, N ), dtype=np.double )
    In     = np.ndarray(shape = ( L+1, N ), dtype=np.double )
    xi     = rand.normal( loc = 0.0, scale = 1.0, size=(L, N))
    V[0]   = ro
    In[0]  = 0.0

    for n in range(L):
        Zero = V[n] == 0.0
        V[n+1]= np.where(Zero,k*th*dt, 0.0) # if V[n] = 0, then V[n+1] = k*th*dt, ow 0

        h   = 1. - exp(-k*dt)
        m   = th + ( V[n] - th)*(1. - h)
        s2  = (s*s*h/k)*( V[n] * (1. - h ) + .5*th*h )
        PSI = s2/(m*m)

        #V[n+1] = 0.0
        Mask   = np.logical_and( PSI > PSI_c, ~Zero )
        u      = rand.uniform(low=0.0, high=1.0, size = N)
        p      = (PSI-1)/(PSI+1)
        opMask = np.logical_and( u > p, Mask == 1 )
        beta   = (1. - p)/m
        x      = np.where(opMask, np.log( (1-p)/(1-u))/beta, 0.0)

        Mask   = np.logical_and( PSI <= PSI_c, ~Zero )
        o      = np.where( Mask, 2/PSI - 1., 0.0)
        b2     = np.where(Mask, o + np.sqrt(o*(o+1)), 0.0) 
        a      = m/(1. + b2)
        c      = np.power( (np.sqrt(b2)+ xi[n]), 2, where=Mask)
        y      = np.where(Mask, a*c, 0)


        V[n+1] += (x + y)
        V[n+1] = np.maximum(V[n+1], 0.0) ######  KEEP VOL >= 0  ######
        In[n+1] = In[n] + (dt/2.) * ( V[n]  + V[n+1] )

    X  = np.ndarray(shape = ( Nt+1, N ), dtype=np.double ) # Y[Nt+1, 2]
    I  = np.ndarray(shape = ( Nt+1, N ), dtype=np.double ) # Y[Nt+1, 2]

    for n in range(Nt+1):
        tn = n * DT
        pos = int(tn/dt)
        X[n] = V[pos]
        I[n] = In[pos]

    return  (X,I)
    

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

def euro_vanilla(rand, So, r, q, T, sigma, k, N, nt, lmbda, nubar, eta, nu_o, rho, dt, NV, NS, mod, eta_vg, nu_vg, theta_vg):
      
    Fw = So*exp((r-q)*T)
    kT = k/Fw

    if mod == "VG":                
        vg = VG(eta=eta_vg, nu=nu_vg, theta=theta_vg)   
        step = T / nt                                       
        S  = VG_trj( rand, So, vg, nt, step, N )
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
    call_cp = put  + exp(-r*T)*Fw*(So - kT)
    put_cp  = call + exp(-r*T)*Fw*(kT - So)

    return  call, put, call_cp, put_cp
# -----------------------------------------------------

def do_mc( **keywrds ):                                                    

    So      = keywrds["So"]
    T       = keywrds["T"]
    nt      = keywrds["nt"]
    r       = keywrds["r"]
    q       = keywrds["q"]
    k       = keywrds["strike"]
    N       = keywrds["N"]
    fp      = keywrds["fp"]
    sigma   = keywrds["sigma"]
    Seed    = keywrds["seed"]
    lmbda   = keywrds["lmbda"]
    nubar   = keywrds["nubar"]
    eta     = keywrds["eta"]
    nu_o    = keywrds["nu_o"]
    rho     = keywrds["rho"]
    dt      = keywrds["dt"]
    NV      = keywrds["NV"]
    NS      = keywrds["NS"]
    mod     = keywrds["mod"]
    eta_vg  = keywrds["eta_vg"]
    nu_vg   = keywrds["nu_vg"]
    theta_vg   = keywrds["theta_vg"]

    rand = np.random.RandomState()
    rand.seed(Seed)
    
    ' MC EURO VANILLA '
    mcCall, mcPut, mcCall_cp, mcPut_cp = euro_vanilla(rand, So, r, q, T, sigma, k, N, nt, lmbda, nubar, eta, nu_o, rho, dt, NV, NS, mod, eta_vg, nu_vg, theta_vg)

    if __name__ == "__main__":
        return stats(mcCall), stats(mcPut), stats(mcCall_cp), stats(mcPut_cp)
    else:
        if k < So*exp((r-q)*T):
            mc, stdc = stats(mcPut)
        elif k >= So*exp((r-q)*T):
            mc, stdc = stats(mcPut_cp)
        
        return mc
# ----------------------------------------------------
    
def run( argv ):

    parms = get_input_parms(argv)

    if "help" in parms:
        usage()
        return
    
    '***********************************************************************'
    '***********************************************************************'
    
    models = ["H", "VG"]

    '****************** ALL *************************'
    So     =  float(parms.get("s", "1.0"))
    k      =  float(parms.get("k", "1.03"))
    sigma  =  float(parms.get("v", ".3"))
    T      =  float(parms.get("T", "1.13"))
    r      =  float(parms.get("r", "0.01"))
    q      =  float(parms.get("q", "0.0"))
    seed   =  int(parms.get("seed", "1"))
    nt     =  int(parms.get("nt", "20"))

    
    '****************** VG ***************************'
    eta_vg   =  float(parms.get("eta_vg", "0.1494"))
    nu_vg    =  float(parms.get("nu_vg", "0.0626"))
    theta_vg =  float(parms.get("theta_vg", "-0.6635"))
    n        =  int(parms.get("n", "22"))


    '*************** HESTON **************************'
    lmbda  =  float(parms.get("lmbda", "7.7648"))
    nubar  =  float(parms.get("nubar", "0.0601"))
    eta    =  float(parms.get("eta", "2.0170"))
    nu_o   =  float(parms.get("nu_o", "0.0475"))
    rho    =  float(parms.get("rho", -0.6952))
    nv     =  int(parms.get("nv", "14"))
    ns     =  int(parms.get("ns", "10"))
    dt     =  float(parms.get("dt", 1./365))

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
    fp.write("\n")
    fp.write("@ %-12s %8.4f\n" %("lambda" , lmbda))
    fp.write("@ %-12s %8.4f\n" %("nubar" , nubar))
    fp.write("@ %-12s %8.4f\n" %("eta" , eta))
    fp.write("@ %-12s %8.4f\n" %("nu_o" , nu_o))
    fp.write("@ %-12s %8.4f\n" %("rho" , rho))
    fp.write("@ %-12s %8.4f\n" %("NV" , NV))
    fp.write("@ %-12s %8.4f\n" %("NS" , NS))
    fp.write("@ %-12s %8.4f\n" %("dt" , dt))
    fp.write("\n")
    fp.write("@ %-12s %8.4f\n" %("eta_vg" , eta_vg))
    fp.write("@ %-12s %8.4f\n" %("nu_vg" , nu_vg))
    fp.write("@ %-12s %8.4f\n" %("theta_vg" , theta_vg))
    fp.write("@ %-12s %8.4f\n" %("N" , N))

    fp.write("\n")
    

    T1 = Timer()
    T1.start()
    
    mc = ["mcCall", "mcPut", "mcCall_cp", "mcPut_cp"]
    analyt = ["Call", "Put"]
    stats = ["m", "std"]
    
    MC_OPT = {x: {MC: {s: [] for s in stats} for MC in mc} for x in models}
    AS_OPT = {x: {a: [] for a in analyt} for x in models}
    
    for model in models:
        
        [MC_OPT[model]["mcCall"]["m"], MC_OPT[model]["mcCall"]["std"]], [MC_OPT[model]["mcPut"]["m"], MC_OPT[model]["mcPut"]["std"]], [MC_OPT[model]["mcCall_cp"]["m"], MC_OPT[model]["mcCall_cp"]["std"]], [MC_OPT[model]["mcPut_cp"]["m"], MC_OPT[model]["mcPut_cp"]["std"]] = do_mc(
            So=So, T=T, r = r, q = q, sigma=sigma, strike=k, N=N, fp=fp, seed=seed, nt=nt, lmbda=lmbda, nubar=nubar, eta=eta, nu_o=nu_o, rho=rho, dt=dt, NV=NV, NS=NS, mod=model, eta_vg=eta_vg, nu_vg=nu_vg, theta_vg=theta_vg)

        [AS_OPT[model]["Call"], AS_OPT[model]["Put"]] = AS(sigma, r, q, So, k, T, lmbda, nubar, eta, nu_o, rho, model, eta_vg, nu_vg, theta_vg)


    end = T1.stop()
    
    
    for model in models:
                
        if model == "VG":
            N_trj = N
        elif model == "H":
            N_trj = NV
        
        fp.write("\n###################### %3s ######################\n" %(model))
        fp.write("\n#-- (S-%5.2f)^+ = %8.5f\n" %(k, AS_OPT[model]["Call"]))
        fp.write("%9s    %10s %3s %12s -- %5s\n" %("N", "E[(S-k)^+]", "+/-", "MC-err", "Op"))
        Op     = ( abs(MC_OPT[model]["mcCall"]["m"]-AS_OPT[model]["Call"]) < 3.*MC_OPT[model]["mcCall"]["std"]/sqrt(N_trj) )
        fp.write("%9d    %10.5f %3s %12.2e -- %5s\n" %( N, MC_OPT[model]["mcCall"]["m"], "+/-", MC_OPT[model]["mcCall"]["std"]/sqrt(N_trj), Op))
        fp.write("\n#----\n")

        # Call via Call-put parity
        fp.write("\n#-- (S-%5.2f)^+ = %8.5f   Call-Put parity\n" %(k, AS_OPT[model]["Call"]))
        fp.write("%9s    %10s %3s %12s -- %5s\n" %("N", "E[(S-k)^+]", "+/-", "MC-err", "Op"))
        Op     = ( abs(MC_OPT[model]["mcCall_cp"]["m"]-AS_OPT[model]["Call"]) < 3.*MC_OPT[model]["mcCall_cp"]["std"]/sqrt(N_trj) )
        fp.write("%9d    %10.5f %3s %12.2e -- %5s\n" %( N, MC_OPT[model]["mcCall_cp"]["m"], "+/-", MC_OPT[model]["mcCall_cp"]["std"]/sqrt(N_trj), Op))
        fp.write("\n#----\n")

        # Standard put
        fp.write("\n#-- (%5.2f - S)^+ = %8.5f\n" %(k, AS_OPT[model]["Put"]))
        fp.write("%9s    %10s %3s %12s -- %5s\n" %("N", "E[(S-k)^+]", "+/-", "MC-err", "Op"))
        Op  = ( abs(MC_OPT[model]["mcPut"]["m"]-AS_OPT[model]["Put"]) < 3.*MC_OPT[model]["mcPut"]["std"]/sqrt(N_trj) )
        fp.write("%9d    %10.5f %3s %12.2e -- %5s\n" %( N, MC_OPT[model]["mcPut"]["m"], "+/-", MC_OPT[model]["mcPut"]["std"]/sqrt(N_trj), Op))
        fp.write("\n#----\n")

        # Put via Call-put parity
        fp.write("\n#-- (%5.2f - S)^+ = %8.5f   Call-Put parity\n" %(k, AS_OPT[model]["Put"]))
        fp.write("%9s    %10s %3s %12s -- %5s\n" %("N", "E[(S-k)^+]", "+/-", "MC-err", "Op"))
        Op  = ( abs(MC_OPT[model]["mcPut_cp"]["m"]-AS_OPT[model]["Put"]) < 3.*MC_OPT[model]["mcPut_cp"]["std"]/sqrt(N_trj) )
        fp.write("%9d    %10.5f %3s %12.2e -- %5s\n" %( N, MC_OPT[model]["mcPut_cp"]["m"], "+/-", MC_OPT[model]["mcPut_cp"]["std"]/sqrt(N_trj), Op))
        fp.write("\n#----\n")
        
        
    fp.write("Elapsed: %10.4f sec.\n" %( end));

    if not name is None: print("@ %-12s: output written to '%s'" %("Info", name))


if __name__ == "__main__":
    run(sys.argv)