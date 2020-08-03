'''
Contains functions to calculate Cl using CAMB in three different ways:
    1) using CAMB's internal function
    2) using CAMB's manual algorithm (uses P_weyl)
    3) using my version of CAMB's manual algorithm (converts P_weyl into P_delta)

Should import CAMB_header
'''

import numpy as np
import camb
from camb import model, initialpower
from .CAMB_header import *

lpa = 10  # lens_potential_accuracy - sweet spot is 10

# Matter Power Spectrum    
def P_delta(z, k, from_func='Weyl'):
    ''' 
    from_func can take 'Weyl', 'psi', 'Weyl linear'
    Defaults to from_func = 'Weyl'
    '''
    if from_func == 'Weyl':
        return 4 / (9 * Omm(z)**2 * H(z)**4) * PK_weyl.P(z, k, grid=False)
    elif from_func == 'psi':
        return 4 / (9 * Omm(z)**2 * H(z)**4 * k**3) * PK_weyl.P(z, k, grid=False)
    if from_func == 'Weyl linear':
        return 4 / (9 * Omm(z)**2 * H(z)**4) * PK_weyl_linear.P(z, k, grid=False)
    else:
        print('Invalid argument for from_func')
        
  
########################################################################################################
#                                   GET CAMB MATTER POWER INTERPOLATORS                                #
########################################################################################################
print('Getting matter power interpolators')

#PK_tot_k = camb.get_matter_power_interpolator(pars, nonlinear=True, hubble_units=False, k_hunit=False, kmax=maxkh, var1=model.Transfer_tot, var2=model.Transfer_tot, zmax=zcmb)

PK_weyl  = camb.get_matter_power_interpolator(pars, nonlinear=True, hubble_units=False, k_hunit=False, kmax=maxkh, var1=model.Transfer_Weyl, var2=model.Transfer_Weyl, zmax=zcmb)

#PK_weyl_linear  = camb.get_matter_power_interpolator(pars, nonlinear=False, hubble_units=False, k_hunit=False, kmax=maxkh, var1=model.Transfer_Weyl, var2=model.Transfer_Weyl, zmax=zcmb)
   

######################################################################################################### 
#                                     AUTOMATIC CAMB CL CALCULATION                                     #
#########################################################################################################

# CAMB internal function
def CAMB_auto(lmax=lmax, save=False, savefolder=None, savename=None):
    '''
    save -> (bool)
    savefolder -> (str)
    savename -> (str)
    
    savefolder example: 'jul08_baryon_analysis/cl_values/'
    savename example: 'cl_camb'
    '''
    l = np.arange(1,lmax+1)
    pars.set_for_lmax(lmax, lens_potential_accuracy=lpa)
    results = camb.get_results(pars)
    cl_camb = results.get_lens_potential_cls(lmax) 
    cl_camb = cl_camb * 2 * np.pi / 4 
    cl_camb = cl_camb[:,0][1:]
    if save==True:
        if type(savefolder)!=str:
            print('INVALID FOLDER: Input a valid savefolder directory!')
            print('Did not save value.')
        elif type(savename)!=str:
            print('INVALID NAME: Input a valid savename!')
            print('Did not save value.')
        else:
            with open('{0}{1}.txt'.format(savefolder, savename), 'w+') as fcl:
                np.savetxt(fcl, cl_camb)
                
            with open('{0}/l.txt'.format(savefolder), 'w+') as fl:
                np.savetxt(fl, l)
                
    return cl_camb

######################################################################################################### 
#                                   MANUAL CAMB CL CALCULATION (WEYL)                                   #
#########################################################################################################
# Cl CAMB manual original using P_weyl
def CAMB_Weyl(save=False, savefolder=None, savename=None):
    '''
    save -> (bool)
    savefolder -> (str)
    savename -> (str)
    
    savefolder example: 'jul08_baryon_analysis/cl_values/'
    savename example: 'cl_camb'
    ''' 
    nz  = 100
    Xs  = np.linspace(0, Xcmb, nz)
    zs  = results.redshift_at_comoving_radial_distance(Xs)
    
    dXs = (Xs[2:]-Xs[:-2])/2
    Xs  = Xs[1:-1]
    zs  = zs[1:-1]
    
    # Get lensing window function (flat universe)
    ws  = ((Xcmb - Xs)/(Xs**2 * Xcmb))**2
    ls  = np.arange(2, lmax+1, dtype=np.float64)
    d   = np.ones(Xs.shape)
    cl_weyl = np.zeros(ls.shape)
    
    for i, li in enumerate(ls):
        k = (li+0.5) / Xs   #kkhc
        d[:] = 1
        d[k<1e-4] = 0
        d[k>=maxkh] = 0
        cl_weyl[i] = np.dot(dXs, d*PK_weyl.P(zs, k, grid=False)*ws / k**4)
    
    cl_weyl = cl_weyl * (ls*(ls+1))**2
    
    if save==True:
        if type(savefolder)!=str:
            print('INVALID FOLDER: Input a valid savefolder directory!')
            print('Did not save value.')
        elif type(savename)!=str:
            print('INVALID NAME: Input a valid savename!')
            print('Did not save value.')
        else:
            with open('{0}{1}.txt'.format(savefolder, savename), 'w+') as fcl_weyl:
                np.savetxt(fcl_weyl, cl_weyl)
                
            with open('{0}/ls.txt'.format(savefolder), 'w+') as fls:
                np.savetxt(fls, ls)
                
    return cl_weyl

######################################################################################################### 
#                                   MANUAL CAMB CL CALCULATION (DELTA)                                  #
#########################################################################################################
# Cl P_weyl -> P_psi
def CAMB_Delta(save=False, savefolder=None, savename=None):
    '''
    save -> (bool)
    savefolder -> (str)
    savename -> (str)
    
    savefolder example: 'jul08_baryon_analysis/cl_values/'
    savename example: 'cl_camb'
    ''' 
    nz  = 100
    Xs  = np.linspace(0, Xcmb, nz)
    zs  = results.redshift_at_comoving_radial_distance(Xs)
    
    dXs = (Xs[2:]-Xs[:-2])/2
    Xs  = Xs[1:-1]
    zs  = zs[1:-1]
    As  = get_a(zs)
    
    # Get lensing window function (flat universe)
    ws = W(Xs, As, Xcmb)
    ls = np.arange(2, lmax+1, dtype=np.float64)
    d  = np.ones(Xs.shape)
    cl_delta = np.zeros(ls.shape)
    
    for i, li in enumerate(ls):
        k = (li+0.5) / Xs
        d[:] = 1
        d[k<1e-4] = 0
        d[k>=maxkh] = 0
        cl_delta[i] = np.dot(dXs, d * P_delta(zs, k, from_func='Weyl') * ws**2 / Xs**2)
        
    if save==True:
        if type(savefolder)!=str:
            print('INVALID FOLDER: Input a valid savefolder directory!')
            print('Did not save value.')
        elif type(savename)!=str:
            print('INVALID NAME: Input a valid savename!')
            print('Did not save value.')
        else:
            with open('{0}{1}.txt'.format(savefolder,savename), 'w+') as fcl_delta:
                np.savetxt(fcl_delta, cl_delta)
                
            with open('{0}/ls.txt'.format(savefolder), 'w+') as fls:
                np.savetxt(fls, ls)
                
    return cl_delta
    

######################################################################################################### 
#                                    CALCULATING BARYONIC CL VALUES                                     #
######################################################################################################### 

def GetBaryonLensingPower(sim_datakey, sim_data, lmax, dl, nz, which_sim='OWLS', save=False, savefolder=None, savename=None):
    '''
    Make sure that if R_int == True, data is data_same
    which_sim takes only 'OWLS','Hz', 'BAHAMAS', 'TNG100' or 'TNG300'
    '''
    
    Xb  = np.linspace(0, Xcmb, nz)
    zb  = results.redshift_at_comoving_radial_distance(Xb)
    if which_sim == 'OWLS':
        kmax = 514.71854     # largest k value from all simulations
        kmin = 0.062831853   # smallest k value from all simulations
    elif which_sim == 'Hz':
        kmax = 32.107
        kmin = 0.07885
    elif (which_sim == 'BAHAMAS') or (which_sim == 'TNG100') or (which_sim == 'TNG300'):
        kmax = 514.71854
        kmin = 0.015707963
        # zmin = 0.000
        # zmax = 3.000
        
    dXb = (Xb[2:]-Xb[:-2])/2
    Xb  = Xb[1:-1]
    zb  = zb[1:-1]
    Ab  = get_a(zb)
    
    # Get lensing window function (flat universe)
    wb = W(Xb, Ab, Xcmb)
    lb = np.arange(10,  lmax+1, dl, dtype=np.float64)
    d  = np.ones(Xb.shape)
    cl_baryon_list = []
    
    # Calculate the integral
    # Use R_interpolator
    for j, datakey in enumerate(sim_datakey):
        print(j, datakey)
        P_ratio_int = sim_data[datakey]['R_interpolator']
        cl_kappa_bary = np.zeros(lb.shape)
        for i, li in enumerate(lb):
            k = (li+0.5) / Xb
            d[:] = 1
            d[k>=kmax] = 0
            cl_kappa_bary[i] = np.dot(dXb, d * P_delta(zb, k, from_func='Weyl') * np.diagonal(np.flip(P_ratio_int(k/(H0/100), zb), axis=1)) * wb**2 / Xb**2) 
        cl_baryon_list.append(cl_kappa_bary)
    
    if save==True:
        if type(savefolder)!=str:
            print('INVALID FOLDER: Input a valid savefolder directory!')
            print('Did not save value.')
        elif type(savename)!=str:
            print('INVALID NAME: Input a valid savename!')
            print('Did not save value.')
        else:
            with open('{0}{1}.txt'.format(savefolder,savename), 'w+') as fcl_bary:
                np.savetxt(fcl_bary, cl_baryon_list)
            with open('{0}/lb.txt'.format(savefolder), 'w+') as flb:
                np.savetxt(flb, lb)
            with open('{0}/{1}-data_key.txt'.format(savefolder, which_sim), 'w+') as fdk:
                        np.savetxt(fdk, sim_datakey, fmt='%s')
    
    return cl_baryon_list

