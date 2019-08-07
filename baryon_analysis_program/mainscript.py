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

import matplotlib.cm as cm
#color = cm.hsv(np.linspace(0, 1.7, len(xs)))
colors = np.array(['r', 'darkorange', 'gold', 'limegreen', 'forestgreen','deepskyblue', 'blue','darkviolet', 'magenta','brown'])

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

# ------------------------------------------------------------------------

# Cl ratio plot with zoom in
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
from matplotlib.patches import Patch

def make_error_boxes(ax, xdata, ydata, xerror, yerror, facecolor='red', edgecolor='None', alpha=0.4):
    # Create list for all the error patches
    errorboxes = []
    # Loop over data points; create box from errors at each point
    for x, y, xe, ye in zip(xdata, ydata, xerror.T, yerror.T):
        rect = Rectangle((x - xe[0], y - ye[0]), xe.sum(), ye.sum())
        errorboxes.append(rect)
    # Create patch collection with specified colour/alpha
    pc = PatchCollection(errorboxes, facecolor=facecolor, alpha=alpha, edgecolor=edgecolor)
    # Add collection to axes
    ax.add_collection(pc)
    # Plot errorbars
    artists = ax.errorbar(xdata, ydata, xerr=xerror, yerr=yerror, fmt='None', ecolor='None')
    return artists

plt.clf()
fig, ax = plt.subplots(1, figsize=(11,8))
for i, datakey in enumerate(data_key): 
    label=datakey.replace('_L100N512','')
    if i == base_index:
        label='DMONLY (DMO)'
    ax.semilogx(lb, clBary[i]/clBary[base_index], color=colors[i], label=label)   # (99991,) --> bin 

leg = ax.legend(ncol=2, loc='upper left', prop={'size': 11})


axins = ax.inset_axes([0.09, 0.25, 0.6, 0.52])
for i, datakey in enumerate(data_key): 
    axins.plot(lb, clBary[i]/clBary[base_index], color=colors[i], label=datakey) 

xdata = xs
ydata = np.ones(xs.shape)       # all ones because Cl_DMO/Cl_DMO
xerr  = SIGMA_F/SIGMA_F *500    # just set as bin
xerr  = np.array([xerr,xerr])
yerr  = SIGMA_F
yerr  = np.array([yerr,yerr])

artist = make_error_boxes(axins, xdata, ydata, xerr, yerr, facecolor='darkviolet')
legend_elements = [Patch(facecolor='darkviolet', edgecolor='None', label='CMB-HD', alpha=0.4)]
legin = ax.legend([artist],handles=legend_elements,loc='upper right', prop={'size': 11}, bbox_to_anchor=(0.6, 0.7, 0.3, 0.3))
ax.add_artist(leg)
ax.add_artist(legin)

axins.set_xlim(10000, 37500)
axins.set_ylim(0.78, 1.22)
axins.grid(True)
axins.tick_params(direction='inout', grid_alpha=0.5, labelsize=11)

ax.indicate_inset_zoom(axins)
ax.tick_params(direction='inout', grid_alpha=0.5, labelsize=12, length=7)

mark_inset(ax, axins, loc1=1, loc2=3, fc='none', ec='0.5')

plt.title(r'Ratio of Baryonic and DMO $C_\ell^{\kappa\kappa}$', size=16)
plt.ylabel(r'$C_\ell^{\kappa\kappa, bary}$/$C_\ell^{\kappa\kappa, DMO}$', size=18)
plt.xlabel(r'$\ell$', size=17)
plt.ylim(0.76)
#plt.show()
filename = savefolder + 'cl_plots/cl_ratio_zoomin_error2.pdf'
plt.savefig(filename, bbox_inches='tight', pad_inches=0.1)


# ------------------------------------------------------------------------

# Cl ratio plot with errors from l = 10000 to 40000
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
from matplotlib.patches import Patch

def make_error_boxes(ax, xdata, ydata, xerror, yerror, facecolor='red', edgecolor='None', alpha=0.4):
    # Create list for all the error patches
    errorboxes = []
    # Loop over data points; create box from errors at each point
    for x, y, xe, ye in zip(xdata, ydata, xerror.T, yerror.T):
        rect = Rectangle((x - xe[0], y - ye[0]), xe.sum(), ye.sum())
        errorboxes.append(rect)
    # Create patch collection with specified colour/alpha
    pc = PatchCollection(errorboxes, facecolor=facecolor, alpha=alpha, edgecolor=edgecolor)
    # Add collection to axes
    ax.add_collection(pc)
    
    # Plot errorbars
    artists = ax.errorbar(xdata, ydata, xerr=xerror, yerr=yerror, fmt='None', ecolor='None')
    legend_elements = [Patch(facecolor=facecolor, edgecolor=edgecolor, label='CMB-HD', alpha=alpha)]
    leg = ax.legend([artists],handles=legend_elements,loc='upper right', prop={'size': 10})
    return artists, leg


# Create figure and axes
plt.clf()
fig, ax = plt.subplots(1, figsize=(10,6))

# Call function to create error boxes
#

for i, datakey in enumerate(data_key): 
    ax.plot(lb, clBary[i]/clBary[base_index], color=colors[i], label=datakey) 

leg1 = ax.legend(ncol=2, loc='upper left', prop={'size': 10})

xdata = xs
ydata = np.ones(xs.shape)       # all ones because Cl_DMO/Cl_DMO
xerr  = SIGMA_F/SIGMA_F *500    # just set as bin
xerr  = np.array([xerr,xerr])
yerr  = SIGMA_F
yerr  = np.array([yerr,yerr])

artist, leg = make_error_boxes(ax, xdata, ydata, xerr, yerr, facecolor='darkviolet')

ax.add_artist(leg1)
ax.add_artist(leg)

plt.xlim(9000,37500)
plt.ylim(0.7, 1.3)
plt.grid(True)

plt.show()


# ------------------------------------------------------------------------

plt.clf()
plt.figure(7, figsize=(10,6))
for i, datakey in enumerate(data_key): 
    plt.semilogx(lb, clBary[i]/clBary[base_index], color=colors[i], label=datakey)   # (99991,) --> bin 
    plt.errorbar(xs, BARY[i](xs)/BARY[base_index](xs), yerr=SIGMA_F, ecolor=colors[i], capsize=4, fmt='none', label='CMB-HD')            # (15,) --> bin

plt.title(r'Ratio $C_\ell^{\kappa\kappa}$ DMONLY vs. $C_\ell^{\kappa\kappa}$ BARYON', size=20)
plt.ylabel(r'$C_\ell^{\kappa\kappa, bary}$/$C_\ell^{\kappa\kappa, DMONLY}$', size=20)
plt.xlabel(r'$\ell$', size=20)
plt.xlim(8000,45000)
plt.grid(True)
plt.legend(ncol=2, loc='upper left', prop={'size': 10})
#plt.show()
plt.savefig('{0}cl_plots/cl_ratio_with_error.pdf'.format(savefolder))
#'''

ending = time.time()
print('Everything took: ', ending - beginning)
