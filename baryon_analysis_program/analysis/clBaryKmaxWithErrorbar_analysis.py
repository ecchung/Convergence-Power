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
colors21 = np.array(['firebrick', 'red', 'coral', 'orange', 'yellow', 'greenyellow', 'limegreen', 'mediumseagreen', 'forestgreen',\
'aqua', 'deepskyblue', 'dodgerblue', 'royalblue', 'mediumblue', 'navy', 'darkslateblue', 'indigo', 'darkviolet', 'magenta', 'deeppink','k'])

######################################################################################################### 
#                                     IMPORTING BARYONIC DATA                                           #
######################################################################################################### 
data_key, data = import_data()
base_key = 'DMONLY_L100N512'
base_index = int(np.argwhere(data_key==base_key))
data_same = interpolate_ratio(data_key, data, base_index)

'''
######################################################################################################### 
#                                        KMAX ANALYSIS PLOT                                             #
######################################################################################################### 

def R(ratio_int, k,z, Kmax):
    """
    k -> (ndarray)
    z -> (ndarray)
    Kmax -> (float)
    """
    if k<=Kmax:
        return ratio_int(k, z)
    else:
        return 1.

R_vec = np.vectorize(R)
                
nz     = 100
cutoff = 514
kmax   = 514.71854                     # largest k value from all simulations
kmin   = 0.062831853                   # smallest k value from all simulations
K_max  = np.linspace(0, 20, 10)
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

#L = np.linspace(10, lmax+1, 800, dtype=np.float64)
L = np.linspace(8000, 42000, 300, dtype=np.float64)

cl_ALLbary_kmax = np.zeros((len(data_key), len(K_max), len(L)))

ind = [0, base_index]

# For each bary sim
for n, datakey in enumerate(data_key):
    print(datakey)
    R_int = data_same[datakey]['R_interpolator']
    
    # For each K_max
    for j, Kmax in enumerate(K_max):
        cl_bary_kmax = np.zeros(L.shape)
        # For each l
        for i, li in enumerate(L):
            k = (li+0.5) / Xs
            d[:] = 1
            d[k>=kmax] = 0 # universal kmax
            cl_bary_kmax[i] = np.dot(dXs, d * P_delta(zs, k, from_func='Weyl') * R_vec(R_int, k, zs, Kmax) * ws**2 / Xs**2)
        # Save for each K_max at nth bary sim
        cl_ALLbary_kmax[n][j] = cl_bary_kmax
                   


# Import errorbar stuff
SIGMA_F = np.loadtxt(savefolder+'cl_values/SIGMA_F.txt')
XS      = np.loadtxt(savefolder+'cl_values/xs.txt')
clBary  = np.loadtxt(savefolder+'cl_values/cl_bary_list_lmax1e5.txt')
lb      = np.loadtxt(savefolder+'cl_values/lb.txt')
BARY    = [] # interpolated cl for all bary
for clbary in clBary:
    BARY.append(interpolate.interp1d(lb,clbary,bounds_error=True))


# Plot Cl difference
cl_ALLbary = cl_ALLbary_kmax
cl_bary_DMONLY = cl_ALLbary_kmax[base_index]

mpl.rcParams.update({'font.family':'serif'})
mpl.rcParams.update({'mathtext.fontset':'cm'})

for n, datakey in enumerate(data_key):
    plt.clf()
    plt.figure(0, figsize=(10,6))
    plt.title(datakey + '\n' + r'$C_\ell$ difference at $k_{max}$ with CMB-HD Error')
    
    for j, Kmax in enumerate(K_max):
        print(j, Kmax)
        cl_bary_kmax = cl_ALLbary_kmax[n][j]
        plt.plot(L, (cl_bary_kmax-cl_bary_DMONLY[j])/cl_bary_DMONLY[j], color=colors11[j], label='{0}'.format(round(Kmax,2)))
        
    plt.errorbar(XS, (BARY[n](XS)-BARY[base_index](XS))/BARY[base_index](XS), yerr=SIGMA_F, ecolor='k', capsize=4, fmt='none', label='CMB-HD') 
    
    ymax = (BARY[n](XS[-2])-BARY[base_index](XS[-2]))/BARY[base_index](XS[-2]) + 1.25*SIGMA_F[-2] #np.mean([SIGMA_F[-1], SIGMA_F[-2]])
    ymin = (BARY[n](XS[-2])-BARY[base_index](XS[-2]))/BARY[base_index](XS[-2]) - 1.25*SIGMA_F[-2]       #np.mean([SIGMA_F[-1], SIGMA_F[-2]])
    plt.legend(title=r'$k_{max}$', ncol=2, fontsize=12, title_fontsize=14)
    plt.ylabel(r'$(C_\ell^{\kappa\kappa, bary} - C_\ell^{\kappa\kappa, DMONLY})/C_\ell^{\kappa\kappa, DMONLY}$', size=16)
    plt.xlabel(r'$\ell$', size=16)
    plt.grid(True, axis='both')
    plt.xlim(10000, 38000)
    plt.ylim(ymin, ymax)
    #plt.show()
    plt.savefig(savefolder+'kmax_analysis/{0}-withErrorbar.pdf'.format(datakey))
''' 
'''   
######################################################################################################### 
#                                    KMAX ANALYSIS % TOLERANCE                                          #
######################################################################################################### 

T = 0.5 #% tolerance
kmax_tol = {}

# Import errorbar stuff
SIGMA_F = np.loadtxt(savefolder+'cl_values/SIGMA_F.txt')
XS      = np.loadtxt(savefolder+'cl_values/xs.txt')
clBary  = np.loadtxt(savefolder+'cl_values/cl_bary_list_lmax1e5.txt')
lb      = np.loadtxt(savefolder+'cl_values/lb.txt')
BARY    = [] # interpolated cl for all bary
for clbary in clBary:
    BARY.append(interpolate.interp1d(lb,clbary,bounds_error=True))



def R(ratio_int, k,z, Kmax):
    """
    k -> (ndarray)
    z -> (ndarray)
    Kmax -> (float)
    """
    if k<=Kmax:
        return ratio_int(k, z)
    else:
        return 1.

R_vec = np.vectorize(R)

                
nz     = 100
cutoff = 514
kmax   = 514.71854                     # largest k value from all simulations
kmin   = 0.062831853                   # smallest k value from all simulations
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

# CHANGE
#L = np.linspace(10, lmax+1, 800, dtype=np.float64)
L = np.linspace(8000, 42000, 300, dtype=np.float64)
K_max  = np.arange(99, 100.1, 0.1)
K_max  = np.append(K_max, kmax)
cl_ALLbary_kmax = np.zeros((len(data_key), len(K_max), len(L)))

ind = [1]# [0, 1, 2, 4] AGN, NOSN, REF, WDENS

# For each bary sim
for n, datakey in enumerate(data_key):
    print(datakey)
    R_int = data_same[datakey]['R_interpolator']
    
    if n in ind:
        # For each K_max
        for j, Kmax in enumerate(K_max):
            cl_bary_kmax = np.zeros(L.shape)
            # For each l
            for i, li in enumerate(L):
                k = (li+0.5) / Xs
                d[:] = 1
                d[k>=kmax] = 0 # universal kmax
                cl_bary_kmax[i] = np.dot(dXs, d * P_delta(zs, k, from_func='Weyl') * R_vec(R_int, k, zs, Kmax) * ws**2 / Xs**2)
            # Save for each K_max at nth bary sim
            cl_ALLbary_kmax[n][j] = cl_bary_kmax


cl = cl_ALLbary_kmax

for n, datakey in enumerate(data_key):
    key = datakey.replace('_L100N512','')
    print('\n' + key)
    kmax_tol[key] = []
    cl_nocut = cl[n][-1]
    flag = 0
    
    if n in ind:
        for j, Kmax in enumerate(K_max[:-1]):
            if flag == 0:
                if max(abs((cl[n][j] - cl_nocut)/cl_nocut))*100 < T:
                    kmax_tol[key].append(Kmax)
                    print(Kmax)
                    flag = 1 # already found the smallest within the tolerance
'''

######################################################################################################### 
#                                   KMAX ANALYSIS CMB-HD TOLERANCE                                      #
######################################################################################################### 

kmax_cmbHD = {}

# Import errorbar stuff
SIGMA   = np.loadtxt(savefolder+'cl_values/SIGMA.txt')
XERR    = np.loadtxt(savefolder+'cl_values/XERR.txt')
XEDGES  = np.loadtxt(savefolder+'cl_values/XEDGES.txt')
XS      = np.loadtxt(savefolder+'cl_values/xs.txt')
clBary  = np.loadtxt(savefolder+'cl_values/cl_bary_list_lmax1e5.txt')
lb      = np.loadtxt(savefolder+'cl_values/lb.txt')
BARY    = [] # interpolated cl for all bary
for clbary in clBary:
    BARY.append(interpolate.interp1d(lb,clbary,bounds_error=True))


def R(ratio_int, k,z, Kmax):
    """
    k -> (ndarray)
    z -> (ndarray)
    Kmax -> (float)
    """
    if k<=Kmax:
        return ratio_int(k, z)
    else:
        return 1.

R_vec = np.vectorize(R)

                
nz     = 100
cutoff = 514
kmax   = 514.71854                     # largest k value from all simulations
kmin   = 0.062831853                   # smallest k value from all simulations
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

# CHANGE
L               = np.linspace(10000, 39000, 400, dtype=np.float64)
K_max           = np.arange(8.0, 9.1, .1)
K_max           = np.append(K_max, kmax)
cl_ALLbary_kmax = np.zeros((len(data_key), len(K_max), len(L)))
YERR            = np.ones(L.shape) # make array of yerrors that correspond to the ell that Cl takes
for i in range(len(SIGMA)-1, -1, -1):
    YERR[L<XEDGES[i+1]] = SIGMA[i]

ind = [4]# [0, 1, 2, 4] AGN, NOSN, REF, WDENS

# For each bary sim
for n, datakey in enumerate(data_key):
    print(datakey)
    R_int = data_same[datakey]['R_interpolator']
    
    if n in ind:
        # For each K_max
        for j, Kmax in enumerate(K_max):
            cl_bary_kmax = np.zeros(L.shape)
            # For each l
            for i, li in enumerate(L):
                k = (li+0.5) / Xs
                d[:] = 1
                d[k>=kmax] = 0 # universal kmax
                cl_bary_kmax[i] = np.dot(dXs, d * P_delta(zs, k, from_func='Weyl') * R_vec(R_int, k, zs, Kmax) * ws**2 / Xs**2)
                    
            # Save for each K_max at nth bary sim
            cl_ALLbary_kmax[n][j] = cl_bary_kmax


cl = cl_ALLbary_kmax

for n, datakey in enumerate(data_key):
    key = datakey.replace('_L100N512','')
    print('\n' + key)
    kmax_cmbHD[key] = []
    cl_nocut = cl[n][-1]
    flag = 0
    
    if n in ind:
        for j, Kmax in enumerate(K_max[:-1]):
            if flag == 0:
                if False in (abs(cl[n][j] - cl_nocut) < YERR):
                    pass
                else:
                    kmax_cmbHD[key].append(Kmax)
                    print(Kmax)
                    flag = 1 # already found the smallest within the tolerance

        
                   




