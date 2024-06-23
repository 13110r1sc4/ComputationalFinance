#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 15:51:53 2024

@author: leonardorisca
"""

' *************************************************************************** '
' *************************** VOLATILITY SURFACE - MC *********************** '
' *************************************************************************** '
' ***** CEV, Displ Diff, Heston ********************************************* '
' *************************************************************************** '

import pyfeng as pf
import pandas as pd
import sys
import numpy as np
from CFLib.timer import Timer
from math import *
from scipy.stats import norm
from CFLib.config import get_input_parms
from sys import stdout as cout
from C_D_H import do_mc
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.interpolate import RegularGridInterpolator
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

def impVolFromMCCall(vPrice, T, vKt):

    scalar = isinstance(vKt, float)
    if scalar: vKt = np.array([vKt])

    vSl = np.zeros(vKt.shape[0])
    vPl = np.maximum(1. - vKt, 0.0) # call

    vSh = np.ones(vKt.shape[0])
    while True:
        vPh = FwEuroCall(T, vSh, vKt)
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
        vPm  = FwEuroCall(T, vSm, vKt)
        mask = vPm > vPrice # if the new price is higher than the price i want,
                            # i will lower the high value for vol
        vSh[mask] = vSm[mask]
        vSl[~mask] = vSm[~mask] # ~ is a not?

    
    if scalar: return .5*(vSh + vSl)[0]
    return .5*(vSh + vSl)
# ------------------------------------

def run_IV_MC( argv ):
    
    parms = get_input_parms(sys.argv)

    if "help" in parms:
      usage()
      return
    
    name   = parms.get("out", None)
    if name is None: fp = cout
    else:            fp = open(name,"w")

    '****************************************************************************'
    '****************************************************************************'
    
    models = ["CEV"] # ["CEV", "DD", "H"]
    
    k_vol_surf = ["0.80","0.85", "0.9", "0.95", "1.", "1.05", "1.1", "1.15", "1.2"]
    
    T_vol_surf_m = ["1", "2", "3", "6", "12", "18"] # months
    
    SIGMA = ["0.6"]
    
    BETA = ["0.55","0.9"]
    
    DELTA = ["0.2", "0.4", "0.5", "0.7"]
    
    '****************** ALL *******************'
    So     =  float(parms.get("s", "1.0"))
    r      =  float(parms.get("r", "0.0"))
    q      =  float(parms.get("q", "0.0"))
    seed   =  int(parms.get("seed", "1"))
    
    '**************** CEV, DD *****************'
    n      =  int(parms.get("n", "20"))    
    nt     =  int(parms.get("nt", "10"))
    
    '*************** HESTON *******************'
    lmbda  =  float(parms.get("lmbda", "7.7648"))
    nubar  =  float(parms.get("nubar", "0.0601"))
    eta    =  float(parms.get("eta", "2.0170"))
    nu_o   =  float(parms.get("nu_o", "0.0475"))
    rho    =  float(parms.get("rho", -0.6952))
    dt    = float(parms.get("dt", 1./365))
    nv     =  int(parms.get("nv", "16"))
    ns     =  int(parms.get("ns", "10"))
    
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
    
    n_months = int(12)
    T_vol_surf_y = list(map(str,[float(T_m)/n_months for T_m in T_vol_surf_m]))
    N = 1 << n
    NS = 1 << ns
    NV = 1 << nv

    
    mcCall  = {x: np.zeros((len(k_vol_surf),len(T_vol_surf_y)))  for x in models}
    IV      = {x: np.zeros((len(k_vol_surf),len(T_vol_surf_y)))  for x in models}
    
    figures = {x: []  for x in models}
    # set up for surface, scatter, scatter_interp
    x = np.vectorize(float)(k_vol_surf)
    y = np.vectorize(float)([round(float(T_y), 3) for T_y in T_vol_surf_y])
    x_interp = np.linspace(np.min(x), np.max(x), num=30)
    y_interp = np.linspace(np.min(y), np.max(y), num=30)
    x_interp, y_interp = np.meshgrid(x_interp, y_interp)
    x_flat = x.flatten()
    y_flat = y.flatten()
    interp_points = np.column_stack((y_interp.flatten(), x_interp.flatten()))
    X, Y = np.meshgrid(x, y)
    x_f = X.flatten()
    y_f = Y.flatten()


    T1 = Timer()
    T1.start()

    for model in models:
                
        mod          =  str(parms.get("mod", model))
        
        if model == "H":
            
            sigma =  float(parms.get("v", "0.0"))
            b     =  float(parms.get("b", "0.0"))
            delta =  float(parms.get("delta", "0.0"))
            
        
            for i_stk, stk in enumerate(k_vol_surf):
                for j_mat, maturity in enumerate(T_vol_surf_y):
                    
                    k      =  float(parms.get("k", stk))
                    T      =  float(parms.get("T", maturity))
            
                    kT = exp((q-r)*T)*k/So
                
                    mcCall[mod][i_stk,j_mat] = do_mc(So=So, T=T, r = r, q = q, b = b, sigma=sigma, strike=k, N=N, fp=fp, seed=seed, nt=nt, delta=delta, lmbda=lmbda, nubar=nubar, eta=eta, nu_o=nu_o, rho=rho, dt=dt, NV=NV, NS=NS,mod=mod)
                    
                    IV[mod][i_stk,j_mat] = impVolFromMCCall(mcCall[mod][i_stk,j_mat], T, k)
                    
            iv = pd.DataFrame(IV[mod][:,:])
            z = iv.transpose()
            surface = go.Surface(x=x, y=y, z=z, colorscale='Viridis')
            z_flat = z.to_numpy().flatten()
            interp_func = RegularGridInterpolator((y_flat, x_flat), z_flat.reshape(z.shape))
            z_interp = interp_func(interp_points)
            z_interp = z_interp.reshape(x_interp.shape)
            scatter_interp = go.Scatter3d(
                x=x_interp.flatten(),
                y=y_interp.flatten(),
                z=z_interp.flatten(),
                mode='markers',
                marker=dict(
                    size=1.2,
                    color='black',
                    opacity=0.8),
                    showlegend=False)
            scatter = go.Scatter3d(
                x=x_f,
                y=y_f,
                z=z_flat,
                mode='markers',
                marker=dict(
                    size=3,
                    color='red',
                    opacity=0.8),
                    showlegend=False)
            
            figures[mod] = go.Figure(data=[surface, scatter_interp, scatter])
            figures[mod].update_layout(
                title="Heston IV", title_x=0.5, title_y=0.9,
                scene=dict(
                    xaxis=dict(gridcolor='black',nticks=10, title='M', tickvals=x),
                    yaxis=dict(gridcolor='black',nticks=4, title='T', tickvals=y),
                    zaxis = dict(nticks=4, title='', range=[iv.min(),iv.max()]),
                    camera=dict(
                         eye=dict(x=1.4, y=-1.7, z=0.9),  # Camera eye position
                         center=dict(x=0, y=-0.25, z=0),  # Point the camera is looking at
                         up=dict(x=0, y=0, z=1))
                    ),
                autosize=True,
                width=500, height=500,
                margin=dict(l=5, r=50, b=30, t=50)) # 65
            figures[mod].write_image("Heston.pdf")
                
                
        elif model == "CEV" or model == "DD":
            
            for j, sgm in enumerate(SIGMA):
                                
                sigma  =  float(parms.get("v", sgm))

                if model == "CEV":
                    
                    delta =  float(parms.get("delta", "0.0"))
                                        
                    for i, bt in enumerate(BETA):
                        
                        b     =  float(parms.get("b", bt))
                        
                        for i_stk, stk in enumerate(k_vol_surf):
                            for j_mat, maturity in enumerate(T_vol_surf_y):
                                
                                k      =  float(parms.get("k", stk))
                                T      =  float(parms.get("T", maturity))
                        
                                kT = exp((q-r)*T)*k/So
                            
                                mcCall[mod][i_stk,j_mat] = do_mc(So=So, T=T, r = r, q = q, b = b, sigma=sigma, strike=k, N=N, fp=fp, seed=seed, nt=nt, delta=delta, lmbda=lmbda, nubar=nubar, eta=eta, nu_o=nu_o, rho=rho, dt=dt, NV=NV, NS=NS,mod=mod)
                                
                                IV[mod][i_stk,j_mat] = impVolFromMCCall(mcCall[mod][i_stk,j_mat], T, k)
                        
                        iv = pd.DataFrame(IV[mod][:,:])
                        z = iv.transpose()
                        surface = go.Surface(x=x, y=y, z=z, colorscale='Viridis')
                        z_flat = z.to_numpy().flatten()
                        interp_func = RegularGridInterpolator((y_flat, x_flat), z_flat.reshape(z.shape))
                        z_interp = interp_func(interp_points)
                        z_interp = z_interp.reshape(x_interp.shape)
                        scatter_interp = go.Scatter3d(
                            x=x_interp.flatten(),
                            y=y_interp.flatten(),
                            z=z_interp.flatten(),
                            mode='markers',
                            marker=dict(
                                size=1.2,
                                color='black',
                                opacity=0.8),
                                showlegend=False)
                        scatter = go.Scatter3d(
                            x=x_f,
                            y=y_f,
                            z=z_flat,
                            mode='markers',
                            marker=dict(
                                size=3,
                                color='red',
                                opacity=0.8),
                                showlegend=False)
                        
                        figures[mod] = go.Figure(data=[surface, scatter_interp, scatter])
                        figures[mod].update_layout(
                            title=f"CEV IV - σ={sgm}, b={bt}", title_x=0.5, title_y=0.9,
                            scene=dict(
                                xaxis=dict(gridcolor='black',nticks=10, title='M', tickvals=x),
                                yaxis=dict(gridcolor='black',nticks=4, title='T', tickvals=y),
                                zaxis = dict(nticks=4, title='', range=[iv.min(),iv.max()]),
                                camera=dict(
                                     eye=dict(x=1.4, y=-1.7, z=0.9),  # Camera eye position
                                     center=dict(x=0, y=-0.25, z=0),  # Point the camera is looking at
                                     up=dict(x=0, y=0, z=1))
                                ),
                            autosize=True,
                            width=500, height=500,
                            margin=dict(l=5, r=50, b=30, t=50)) # 65
                        figures[mod].write_image(f"CEV_{i}{j}.pdf")

                elif model == "DD":
                    
                    b     =  float(parms.get("b", "0.0"))
                    
                    for i, dlt in enumerate(DELTA):
                        
                        delta =  float(parms.get("delta", dlt))
                        
                        for i_stk, stk in enumerate(k_vol_surf):
                            for j_mat, maturity in enumerate(T_vol_surf_y):
                                
                                k      =  float(parms.get("k", stk))
                                T      =  float(parms.get("T", maturity))
                        
                                kT = exp((q-r)*T)*k/So
                            
                                mcCall[mod][i_stk,j_mat] = do_mc(So=So, T=T, r = r, q = q, b = b, sigma=sigma, strike=k, N=N, fp=fp, seed=seed, nt=nt, delta=delta, lmbda=lmbda, nubar=nubar, eta=eta, nu_o=nu_o, rho=rho, dt=dt, NV=NV, NS=NS,mod=mod)
                                
                                IV[mod][i_stk,j_mat] = impVolFromMCCall(mcCall[mod][i_stk,j_mat], T, k)
                                
                        iv = pd.DataFrame(IV[mod][:,:])
                        z = iv.transpose()
                        surface = go.Surface(x=x, y=y, z=z, colorscale='Viridis')
                        z_flat = z.to_numpy().flatten()
                        interp_func = RegularGridInterpolator((y_flat, x_flat), z_flat.reshape(z.shape))
                        z_interp = interp_func(interp_points)
                        z_interp = z_interp.reshape(x_interp.shape)
                        scatter_interp = go.Scatter3d(
                            x=x_interp.flatten(),
                            y=y_interp.flatten(),
                            z=z_interp.flatten(),
                            mode='markers',
                            marker=dict(
                                size=1.2,
                                color='black',
                                opacity=0.8),
                                showlegend=False)
                        scatter = go.Scatter3d(
                            x=x_f,
                            y=y_f,
                            z=z_flat,
                            mode='markers',
                            marker=dict(
                                size=3,
                                color='red',
                                opacity=0.8),
                                showlegend=False)
                        
                        figures[mod] = go.Figure(data=[surface, scatter_interp, scatter])
                        figures[mod].update_layout(
                            title=f"DD IV - σ={sgm}, Δ={dlt}", title_x=0.5, title_y=0.9,
                            scene=dict(
                                xaxis=dict(gridcolor='black',nticks=10, title='M', tickvals=x),
                                yaxis=dict(gridcolor='black',nticks=4, title='T', tickvals=y),
                                zaxis = dict(nticks=4, title='', range=[iv.min(),iv.max()]),
                                camera=dict(
                                     eye=dict(x=1.4, y=-1.7, z=0.9),  # Camera eye position
                                     center=dict(x=0, y=-0.25, z=0),  # Point the camera is looking at
                                     up=dict(x=0, y=0, z=1))
                                ),
                            autosize=True,
                            width=500, height=500,
                            margin=dict(l=5, r=50, b=30, t=50)) # 65
                        figures[mod].write_image(f"DD_{i}{j}.pdf")

    end = T1.stop()
    fp.write("@ %-12s %8.4f\n" %("end" , end))

if __name__ == "__main__":
    run_IV_MC( sys.argv )


