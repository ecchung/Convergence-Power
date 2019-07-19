# BARYONIC ANALYSIS (abstract version of baryon_analysis_old.py)
# Last edited July 17, 2019
import time
beginning = time.time()

import numpy as np
import matplotlib.pyplot as plt
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
cl_camb  = CAMB_auto(save=True, savefolder=savefolder+'cl_values/', savename='cl_camb')
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
cl_camb  = np.loadtxt('{0}cl_values/cl_camb.txt'.format(savefolder))
cl_weyl  = np.loadtxt('{0}cl_values/cl_weyl.txt'.format(savefolder))
cl_delta = np.loadtxt('{0}cl_values/cl_delta.txt'.format(savefolder))   
l        = np.loadtxt('{0}cl_values/l.txt'.format(savefolder))  

                                #-----------BARYONIC CL-----------# 
cl_baryon_list         = np.loadtxt('{0}cl_values/cl_bary_list.txt'.format(savefolder))
cl_baryon_list_lmax1e5 = np.loadtxt('{0}cl_values/cl_bary_list_lmax1e5.txt'.format(savefolder))
lb                     = np.loadtxt('{0}cl_values/lb.txt'.format(savefolder))


# -------------------------------------------------------------------------------------------------------

######################################################################################################### 
#                                           ERROR ANALYSIS                                              #
######################################################################################################### 
                                #--------IMPORT COV. MATRIX-------# 
covmat    = np.load('covmat.npy')
covinv    = inv(covmat)
sigma     = np.sqrt((covmat.diagonal()))
lBinEdges = np.load('lbin_edges.npy')

                                #----------CALC DELTA CHI----------#                                
'''
clBinned  = np.zeros((len(data_key), len(lBinEdges)-1))  # average the cl values within the bins 
lMidBin   = np.zeros(len(lBinEdges)-1)  # values of l in the middle of the bin
d         = np.ones(lb.shape)

for j, clBaryon in enumerate(cl_baryon_list_lmax1e5):
    cl = clBaryon
    for i in range(len(lBinEdges)-1):
        
        if j == 0: # just do this once
            lMidBin[i] = (lBinEdges[i] + lBinEdges[i+1]) / 2
        
        d[:] = 1  # reset d to all 1
        d[lb<lBinEdges[i]] = 0
        d[lb>=lBinEdges[i+1]] = 0
        cl = clBaryon  # reset cl to original clBaryon
        cl = cl*d
        
        clBinMean = sum(cl) / sum(d)
        clBinned[j][i] = clBinMean # save to ith bin of jth simulation

# Calculate delta chi squared for each simulation
DeltaChi2 = np.zeros(data_key.shape)
clDMO = clBinned[base_index]
for i, clBary in enumerate(clBinned):
    DeltaChi2[i] = sum(((clBary-clDMO)/sigma)**2)
    
# Save values
with open('{0}{1}.txt'.format(savefolder+'cl_values/','clBinned'), 'w+') as fclbin:
    np.savetxt(fclbin, clBinned)

with open('{0}{1}.txt'.format(savefolder+'cl_values/','lMidBin'), 'w+') as flmb:
    np.savetxt(flmb, lMidBin)

with open('{0}{1}.txt'.format(savefolder+'cl_values/','DeltaChi2'), 'w+') as fdc:
    np.savetxt(fdc, DeltaChi2)
'''    
# Import values
clMeanBin = np.loadtxt('{0}cl_values/clBinned.txt'.format(savefolder))
lMidBin   = np.loadtxt('{0}cl_values/lMidBin.txt'.format(savefolder))
DeltaChi2 = np.loadtxt('{0}cl_values/DeltaChi2.txt'.format(savefolder))

                                #--------DELTA CHI ANALYSIS--------# 
clBary   = cl_baryon_list_lmax1e5
clMidBin = np.zeros((len(data_key), len(lMidBin)))

# Extract the Cl values at ell MidBin
for j, l in enumerate(lMidBin):
    for i in range(len(data_key)):
        clMidBin[i,j] = clBary[i,int(lMidBin[j]-10)]

sigmaFrac = sigma/clMidBin[base_index] # fractional sigma

                                #-------PLOT WITH DELTA CHI-------# 
plt.figure(7, figsize=(10,6))
plt.clf()

for i, datakey in enumerate(data_key): 
    plt.semilogx(lb, (clBary[i]-clBary[base_index])/clBary[base_index], color=colors[i], label=datakey)    

plt.errorbar(lMidBin, clMidBin[base_index], yerr=sigmaFrac, ecolor='k')
plt.title(r'Difference $C_\ell^{\kappa\kappa}$ DMONLY vs. $C_\ell^{\kappa\kappa}$ BARYON')
plt.ylabel(r'($C_\ell^{\kappa\kappa, bary}$ - $C_\ell^{\kappa\kappa, DMONLY}$)/$C_\ell^{\kappa\kappa, DMONLY}$')
plt.xlabel(r'Multipole $\ell$')
plt.grid(True)
plt.legend(ncol=2, loc='upper left', prop={'size': 10})
plt.show

#plt.savefig('{0}cl_plots/cl_diff_lmax={1}.pdf'.format(savefolder, lmax))



ending = time.time()
print('Everything took: ', ending - beginning)