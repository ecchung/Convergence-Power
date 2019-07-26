# BARYONIC ANALYSIS (abstract version of baryon_analysis_old.py)
# Last edited July 17, 2019
import time
beginning = time.time()

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import camb
from camb import model, initialpower
from scipy.constants import c           # c / 1000 = 299 km/s
import scipy.integrate as integrate
from scipy import interpolate
from numpy.linalg import inv
from header.CAMB_header import *
from header.calculate_cl import CAMB_auto, CAMB_Weyl, CAMB_Delta, P_delta, calc_cl_bary
from header.import_baryon_data import import_data, interpolate_ratio

savefolder = 'outputs_jul17/'
datafolder = 'baryonic_data'

colors = np.array(['r', 'darkorange', 'yellow', 'limegreen', 'forestgreen','deepskyblue', 'blue','darkviolet', 'magenta','brown'])

######################################################################################################### 
#                                     IMPORTING BARYONIC DATA                                           #
######################################################################################################### 
data_key, data = import_data()
base_key = 'DMONLY_L100N512'
base_index = int(np.argwhere(data_key==base_key))
data_same = interpolate_ratio(data_key, data, base_index)

'''
######################################################################################################### 
#                                    CALCULATE & SAVE CL VALUES                                         #
######################################################################################################### 
                                #-------------CAMB CL-------------# 
cl_camb  = CAMB_auto(lmax=40000, save=True, savefolder=savefolder+'cl_values/', savename='cl_camb')
cl_weyl  = CAMB_Weyl(save=True, savefolder=savefolder+'cl_values/', savename='cl_weyl')
cl_delta = CAMB_Delta(save=True, savefolder=savefolder+'cl_values/', savename='cl_delta')

                                #-----------BARYONIC CL-----------# 
nz = 100
lmax = 1e5
dl = 1
cl_baryon_list = calc_cl_bary(data_key, data_same, base_index, lmax, dl, nz, save=True, savefolder=savefolder+'cl_values/', savename='cl_bary_list_lmax1e5', R_int=True)
'''

######################################################################################################### 
#                                         IMPORT CL VALUES                                              #
######################################################################################################### 
                                #-------------CAMB CL-------------# 
cl_camb  = np.loadtxt('{0}cl_values/cl_camb.txt'.format(savefolder))    # l from 0 to 40000
cl_weyl  = np.loadtxt('{0}cl_values/cl_weyl.txt'.format(savefolder))    # l from 0 to 5000
cl_delta = np.loadtxt('{0}cl_values/cl_delta.txt'.format(savefolder))   # l from 0 to 5000
l        = np.loadtxt('{0}cl_values/l.txt'.format(savefolder))  # currently range from 0 to 40000

                                #-----------BARYONIC CL-----------# 
cl_baryon_list         = np.loadtxt('{0}cl_values/cl_bary_list.txt'.format(savefolder))
cl_baryon_list_lmax1e5 = np.loadtxt('{0}cl_values/cl_bary_list_lmax1e5.txt'.format(savefolder))
lb                     = np.loadtxt('{0}cl_values/lb.txt'.format(savefolder))


# -------------------------------------------------------------------------------------------------------

######################################################################################################### 
#                                           ERROR ANALYSIS                                              #
######################################################################################################### 
fsky = 0.5
nbin = 15
Lrange = np.linspace(10000,40000,nbin)

                                #--------IMPORT COV. MATRIX-------# 
covmat    = np.load('error_data/covmat.npy') # (129, 129) --> li, li
covInv    = inv(covmat)                      # (129, 129) --> li, li
covDiag   = covmat.diagonal() * (2.912603855229585/41253.) / fsky # (129, ) --> li  KEEP!
lBinEdges = np.load('error_data/lbin_edges.npy')                  # (130, ) --> li + 1
lMidBin   = (lBinEdges[1:] + lBinEdges[:-1]) / 2                  # (129, ) --> li

clBary   = cl_baryon_list_lmax1e5   # (10, 99991) --> data_key, l
BARY = [] # interpolated cl for all bary
for clbary in clBary:
    BARY.append(interpolate.interp1d(lb,clbary,bounds_error=True))
    
                                #--------FANCY BINNED ERRORS-------# 
                                #       FROM NAM NGUYEN'S CODE     #
                                #----------------------------------#           
xs,ys = [],[]
x,y = 0.,0.
j = 0
count = 0

for i in range(len(covDiag)):
    if Lrange[j] < lMidBin[i]: # end of bin
        if count != 0:
            xs.append(x/count)
            ys.append(1./np.sqrt(y))
            count = 0
        j += 1
        x,y = 0.,0.
    count += 1
    x += lMidBin[i]
    y += 1./covDiag[i]

if count != 0:
    xs.append(x/count)
    ys.append(1./np.sqrt(y))

xs = np.array(xs)
ys = np.array(ys) # y error



                                #----S/N + DELTA CHI SQ ANALYSIS---# 
                                
SN    = []    # (10, ) --> data_key
DCHI2 = []    # (10, ) --> data_key

for bary in BARY: 
    DCHI2.append(sum( (bary(xs)-BARY[base_index](xs)) / ys)**2 )
    SN.append(np.sqrt(sum( (bary(xs)/ys)**2) ))

SN    = np.array(SN)
DCHI2 = np.array(DCHI2)

with open('{0}{1}.txt'.format(savefolder+'cl_values/','SN'), 'w+') as fs:
    np.savetxt(fs, SN)
with open('{0}{1}.txt'.format(savefolder+'cl_values/','DCHI2'), 'w+') as fdc:
    np.savetxt(fdc, DCHI2)
    
SIGMA_F = ys/BARY[base_index](xs)  # (15, ) --> bin # fractional sigma
SIGMA = ys

with open('{0}{1}.txt'.format(savefolder+'cl_values/','SIGMA_F'), 'w+') as fsigf:
    np.savetxt(fsigf, SIGMA_F)

with open('{0}{1}.txt'.format(savefolder+'cl_values/','SIGMA'), 'w+') as fsig:
    np.savetxt(fsig, SIGMA)

with open('{0}{1}.txt'.format(savefolder+'cl_values/','xs'), 'w+') as fxs:
    np.savetxt(fxs, xs)


#'''
                                #----------PLOT WITH ERROR---------# 
mpl.rcParams.update({'font.family':'serif'})
mpl.rcParams.update({'mathtext.fontset':'cm'})

plt.clf()
plt.figure(7, figsize=(10,6))
for i, datakey in enumerate(data_key): 
    plt.semilogy(lb, clBary[i], color=colors[i], label=datakey)   # (99991,) --> bin 
    plt.errorbar(xs, BARY[i](xs), yerr=SIGMA, ecolor=colors[i], capsize=4, fmt='none')            # (15,) --> bin

plt.semilogy(l, cl_camb, color='k', label='cl CAMB no baryon')
plt.title(r'$C_\ell^{\kappa\kappa}$ BARYON', size=20)
plt.ylabel(r'$C_\ell^{\kappa\kappa, bary}$', size=20)
plt.xlabel(r'$\ell$', size=20)
plt.xlim(10000,38000)
plt.ylim(1.3e-11,2.2e-10)
plt.grid(True)
plt.legend(ncol=2, loc='upper right', prop={'size': 10})
#plt.show()
plt.savefig('{0}cl_plots/cl_with_error.pdf'.format(savefolder))



plt.clf()
plt.figure(7, figsize=(10,6))
for i, datakey in enumerate(data_key): 
    plt.semilogx(lb, (clBary[i]-clBary[base_index])/clBary[base_index], color=colors[i], label=datakey)   # (99991,) --> bin 
    plt.errorbar(xs, (BARY[i](xs)-BARY[base_index](xs))/BARY[base_index](xs), yerr=SIGMA_F, ecolor=colors[i], capsize=4, fmt='none', label='CMB-HD')            # (15,) --> bin

plt.title(r'Difference $C_\ell^{\kappa\kappa}$ DMONLY vs. $C_\ell^{\kappa\kappa}$ BARYON', size=20)
plt.ylabel(r'($C_\ell^{\kappa\kappa, bary}$ - $C_\ell^{\kappa\kappa, DMONLY}$)/$C_\ell^{\kappa\kappa, DMONLY}$', size=20)
plt.xlabel(r'$\ell$', size=20)
plt.xlim(8000,45000)
plt.grid(True)
plt.legend(ncol=2, loc='upper left', prop={'size': 10})
#plt.show()
plt.savefig('{0}cl_plots/cl_diff_with_error.pdf'.format(savefolder))
#'''

ending = time.time()
print('Everything took: ', ending - beginning)