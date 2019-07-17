import numpy as np
import matplotlib.pyplot as plt
import camb
from camb import model, initialpower
from scipy.constants import c           # c / 1000 = 299 km/s
import scipy.integrate as integrate
from scipy import interpolate

import sys, os
sys.path.append(os.path.abspath('..')) # now in baryon_analysis_program folder
dirpath = os.getcwd() 

from plot_scripts.plot_exp_offset import bottom_offset, top_offset
import types
from header.CAMB_header import *
from header.calculate_cl import P_delta
from header.import_baryon_data import import_data, interpolate_ratio


savefolder = dirpath+'/../outputs_jul17/kmax_analysis/'
datafolder = 'baryonic_data'

colors = np.array(['r', 'darkorange', 'yellow', 'limegreen', 'forestgreen','deepskyblue', 'blue','darkviolet', 'magenta','brown'])

######################################################################################################### 
#                                     IMPORTING BARYONIC DATA                                           #
######################################################################################################### 
data_key, data = import_data()
base_key = 'DMONLY_L100N512'
base_index = int(np.argwhere(data_key==base_key))
data_same = interpolate_ratio(data_key, data, base_index)


######################################################################################################### 
#                                          KMAX ANALYSIS                                                #
######################################################################################################### 

def R(ratio_int, k,z, Kmax):
    '''
    k -> (ndarray)
    z -> (ndarray)
    Kmax -> (float)
    '''
    if k<=Kmax:
        return ratio_int(k, z)
    else:
        return 1.

R_vec = np.vectorize(R)
                
nz     = 100
cutoff = 514
kmax   = 514.71854                     # largest k value from all simulations
kmin   = 0.062831853                   # smallest k value from all simulations
K_max  = np.linspace(kmax-cutoff, kmax, 10)      # varying kmax
K_max  = np.flip(K_max)
K_max  = np.linspace(0, 50, 9)
K_max  = np.append(K_max, kmax)
lmax   = 1e5

Xs  = np.linspace(0, Xcmb, nz)
zs  = results.redshift_at_comoving_radial_distance(Xs)

dXs = (Xs[2:]-Xs[:-2])/2
Xs  = Xs[1:-1]
zs  = zs[1:-1]
As  = get_a(zs)
ws  = W(Xs, As, Xcmb)
ls  = np.arange(2, lmax+1, dtype=np.float64)
d   = np.ones(Xs.shape)

L = np.linspace(10, lmax+1, 200, dtype=np.float64)


cl_ALLbary_kmax = np.zeros((len(data_key), len(K_max), len(L)))
cl_ALLbary_norm = np.zeros((len(data_key), len(K_max), len(L)))

# For each bary sim
for n, datakey in enumerate(data_key):
    print(datakey)
    R_int = data_same[datakey]['R_interpolator']
    
    # For each K_max
    for j, Kmax in enumerate(K_max):
        cl_bary_kmax = np.zeros(L.shape)
        cl_bary_norm = np.zeros(L.shape)
        # For each l
        for i, li in enumerate(L):
            k = (li+0.5) / Xs
            d[:] = 1
            d[k>=kmax] = 0 # universal kmax
            cl_bary_kmax[i] = np.dot(dXs, d * P_delta(zs, k, from_func='Weyl') * R_vec(R_int, k, zs, Kmax) * ws**2 / Xs**2)
            cl_bary_norm[i] = np.dot(dXs, d * P_delta(zs, k, from_func='Weyl') * np.diagonal(np.flip(R_int(k, zs), axis=1)) * ws**2 / Xs**2) 
        # Save for each K_max at nth bary sim
        cl_ALLbary_kmax[n][j] = cl_bary_kmax
        cl_ALLbary_norm[n][j] = cl_bary_norm
                    

# Plot Cl difference
cl_ALLbary = cl_ALLbary_kmax

cl_bary_DMONLY = cl_ALLbary[base_index]
for n, datakey in enumerate(data_key):
    plt.clf()
    plt.figure(n, figsize=(10,6))
    plt.title(datakey + '\n' + r'$C_\ell$ difference at $k_{max}$')
    
    for j, Kmax in enumerate(K_max):
        print(j)
        cl_bary_kmax = cl_ALLbary_kmax[n][j]
        cl_bary_norm = cl_ALLbary_norm[n][j]
        plt.semilogx(L, (cl_bary_kmax-cl_bary_DMONLY[j])/cl_bary_DMONLY[j], color=colors[j], label=r'$k_{max}$' + '= {0}'.format(round(Kmax,2)))
        plt.semilogx(L, (cl_bary_norm-cl_bary_DMONLY[j])/cl_bary_DMONLY[j], color=colors[j], label=r'$k_{max}$' + '= {0}'.format(round(Kmax,2)), ls='--')
        
    plt.legend(ncol=2)
    plt.ylabel(r'$(C_\ell^{\kappa\kappa, bary} - C_\ell^{\kappa\kappa, DMONLY})/C_\ell^{\kappa\kappa, DMONLY}$')
    plt.xlabel(r'$\ell$')
    plt.grid(True)
    plt.ylim(-0.2)
    #plt.show()
    plt.savefig(savefolder+'{0}.pdf'.format(datakey))
    
    