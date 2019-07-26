import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
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


savefolder = dirpath+'/../outputs_jul17/'
datafolder = 'baryonic_data'

colors11 = np.array(['r', 'darkorange', 'yellow', 'limegreen', 'forestgreen','deepskyblue', 'blue','darkviolet', 'magenta', 'brown','k'])
colors16 = np.array(['firebrick','red', 'coral', 'orange', 'yellow', 'greenyellow', 'limegreen','forestgreen','aqua', 'deepskyblue', 'dodgerblue', 'mediumblue','darkviolet', 'magenta', 'deeppink','k'])
colors21 = np.array(['firebrick', 'red', 'coral', 'orange', 'yellow', 'greenyellow', 'limegreen', 'mediumseagreen', 'forestgreen',\
'aqua', 'deepskyblue', 'dodgerblue', 'royalblue', 'mediumblue', 'navy', 'darkslateblue', 'indigo', 'darkviolet', 'magenta', 'deeppink','k'])

######################################################################################################### 
#                                     IMPORTING BARYONIC DATA                                           #
######################################################################################################### 
data_key, data = import_data()
base_key       = 'DMONLY_L100N512'
base_index     = int(np.argwhere(data_key==base_key))
data_same      = interpolate_ratio(data_key, data, base_index)
z_same         = data_same[data_key[0]]['z']

######################################################################################################### 
#                                          ZMAX ANALYSIS                                                #
######################################################################################################### 

def R(ratio_int, k,z, zmax):
    '''
    k -> (ndarray)
    z -> (ndarray)
    zmax -> (float)
    '''
    if z<zmax:
        return ratio_int(k, z)
    else:
        return 1

R_vec = np.vectorize(R)


# Varying the z_max
nz  = 100
Xb  = np.linspace(0, Xcmb, nz)      # full range
zb  = results.redshift_at_comoving_radial_distance(Xb)
ZMAX = np.linspace(0, 4, 15) #np.flip(z_same[:10])
ZMAX = np.append(ZMAX, z_same[-1])

kmax = 514.71854                     # largest k value from all simulations
kmin = 0.062831853                   # smallest k value from all simulations
lmax = 1e5

dXb = (Xb[2:]-Xb[:-2])/2
Xb  = Xb[1:-1]
zb  = zb[1:-1]
Ab  = get_a(zb)
wb = W(Xb, Ab, Xcmb)
d  = np.ones(Xb.shape)

#L = np.arange(10, lmax+1, 500, dtype=np.float64)
L = np.linspace(8000, 42000, 300, dtype=np.float64)
   
cl_ALLbary_zmax = np.zeros((len(data_key), len(ZMAX), len(L)))
    
ind = [0, base_index]
    
# For each bary sim
for n, datakey in enumerate(data_key):
    print(datakey)
    R_int = data_same[datakey]['R_interpolator']
    
    # For each ZMAX
    for j, zmax in enumerate(ZMAX):
        cl_bary_zmax = np.zeros(L.shape)
        # For each l
        for i, li in enumerate(L):
            k = (li+0.5) / Xb
            d[:] = 1
            d[k>=kmax] = 0 # universal kmax
            cl_bary_zmax[i] = np.dot(dXb, d * P_delta(zb, k, from_func='Weyl') * R_vec(R_int, k,zb, zmax) * wb**2 / Xb**2)
        # Save for each K_max at nth bary sim
        cl_ALLbary_zmax[n][j] = cl_bary_zmax


# Import errorbar stuff
SIGMA_F = np.loadtxt(savefolder+'cl_values/SIGMA_F.txt')
XS      = np.loadtxt(savefolder+'cl_values/xs.txt')
clBary  = np.loadtxt(savefolder+'cl_values/cl_bary_list_lmax1e5.txt')
lb      = np.loadtxt(savefolder+'cl_values/lb.txt')
BARY    = [] # interpolated cl for all bary
for clbary in clBary:
    BARY.append(interpolate.interp1d(lb,clbary,bounds_error=True))


# Plot Cl difference
cl_ALLbary = cl_ALLbary_zmax
cl_DMONLY  = cl_ALLbary_zmax[base_index]

mpl.rcParams.update({'font.family':'serif'})
mpl.rcParams.update({'mathtext.fontset':'cm'})

for n, datakey in enumerate(data_key):
    plt.clf()
    plt.figure(0, figsize=(10,8))
    plt.title(datakey + '\n' + r'$C_\ell$ difference at $z_{max}$ with CMB-HD Error')
    
    for j, zmax in enumerate(ZMAX):
        print(j, zmax)
        cl_zmax = cl_ALLbary[n][j]
        plt.plot(L, (cl_zmax-cl_DMONLY[j])/cl_DMONLY[j], color=colors16[j], label='{0}'.format(round(zmax,2)))
        
    plt.errorbar(XS, (BARY[n](XS)-BARY[base_index](XS))/BARY[base_index](XS), yerr=SIGMA_F, ecolor='k', capsize=4, fmt='none', label='CMB-HD') 
    
    ymax = (BARY[n](XS[-2])-BARY[base_index](XS[-2]))/BARY[base_index](XS[-2]) + 1.05*SIGMA_F[-2] 
    ymin = (BARY[n](XS[-2])-BARY[base_index](XS[-2]))/BARY[base_index](XS[-2]) - 1.05*SIGMA_F[-2]  
    plt.legend(title=r'$z_{max}$', ncol=3, fontsize=12, title_fontsize=14)
    plt.ylabel(r'$(C_\ell^{\kappa\kappa, bary} - C_\ell^{\kappa\kappa, DMONLY})/C_\ell^{\kappa\kappa, DMONLY}$', size=16)
    plt.xlabel(r'$\ell$', size=16)
    plt.grid(True, axis='both')
    plt.xlim(10000, 38000)
    plt.ylim(ymin, ymax)
    #plt.show()
    plt.savefig(savefolder+'zmax_analysis/{0}-withErrorbar.pdf'.format(datakey))
    
    




