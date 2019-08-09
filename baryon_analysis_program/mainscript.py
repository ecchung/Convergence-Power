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
from header.calculate_cl import CAMB_auto, CAMB_Weyl, CAMB_Delta, P_delta, GetBaryonLensingPower
from header.import_baryon_data import import_data, interpolate_ratio

savefolder = 'outputs_jul17/'
datafolder = 'baryonic_data'

import matplotlib.cm as cm
#color = cm.hsv(np.linspace(0, 1.7, len(xs)))
colors = np.array(['r', 'darkorange', 'gold', 'limegreen', 'forestgreen','deepskyblue', 'blue','darkviolet', 'fuchsia','brown'])

######################################################################################################### 
#                                     IMPORTING BARYONIC DATA                                           #
######################################################################################################### 
import copy

data_key, data = import_data()

OWLS_base_key  = 'DMONLY_L100N512'
OWLS_base_index= int(np.argwhere(data_key==OWLS_base_key)) # OWLS
OWLS_datakey = []
Hz_datakey = []
for i, key in enumerate(data_key):
    if 'Hz' in key:
        Hz_datakey.append(key)
    else:
        OWLS_datakey.append(key)

OWLS_datakey = np.array(OWLS_datakey)
OWLS_data    = interpolate_ratio(OWLS_datakey, data, OWLS_base_index) #for OWLS only
data_same    = OWLS_data
base_index   = OWLS_base_index

Hz_datakey    = np.array(Hz_datakey)
Hz_base_key   = 'Hz-DM'
Hz_base_index = int(np.argwhere(data_key==Hz_base_key))   # Horizon
Hz_data, Hz_data_fix, Hz_data_nofix = copy.deepcopy(data),copy.deepcopy(data),copy.deepcopy(data)

Hz_data_nofix = interpolate_ratio_Hz(data_key, Hz_data_nofix, fix=False)
Hz_data       = interpolate_ratio_Hz(data_key, Hz_data, fix='Maybe')
Hz_data_fix   = interpolate_ratio_Hz(data_key, Hz_data_fix, fix=True)

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
cl_OWLS_list = GetBaryonLensingPower(OWLS_datakey, OWLS_data, OWLS_base_index, lmax, dl, nz, which_sim='OWLS', save=True, savefolder=savefolder+'cl_values/', savename='cl_bary_list_lmax1e5')
cl_Hz_list = GetBaryonLensingPower(Hz_datakey, Hz_data, Hz_base_index, lmax, dl, nz, which_sim='Hz', save=True, savefolder=savefolder+'cl_values/', savename='cl_Hz_list_lmax1e5')
cl_Hz_nofix_list = GetBaryonLensingPower(Hz_datakey, Hz_data_nofix, Hz_base_index, lmax, dl, nz, which_sim='Hz', save=True, savefolder=savefolder+'cl_values/', savename='cl_Hz_nofix_list_lmax1e5')
cl_Hz_fix_list = GetBaryonLensingPower(Hz_datakey, Hz_data_fix, Hz_base_index, lmax, dl, nz, which_sim='Hz', save=True, savefolder=savefolder+'cl_values/', savename='cl_Hz_fix_list_lmax1e5')

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
cl_Hz_list             = np.loadtxt('{0}cl_values/cl_Hz_list_lmax1e5.txt'.format(savefolder))
cl_Hz_nofix_list       = np.loadtxt('{0}cl_values/cl_Hz_nofix_list_lmax1e5.txt'.format(savefolder))
cl_Hz_fix_list         = np.loadtxt('{0}cl_values/cl_Hz_fix_list_lmax1e5.txt'.format(savefolder))

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
xsEdges = [300] # first number of lBinEdges
x,y = 0.,0.
j = 0
count = 0

for i in range(len(covDiag)):
    if Lrange[j] < lMidBin[i]: # end of bin
        print(Lrange[j], i, lMidBin[i], lBinEdges[i])
        xsEdges.append(lBinEdges[i])
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
    xsEdges.append(lBinEdges[-1])

xs = np.array(xs)
ys = np.array(ys) # y error
xsLeft = xs - xsEdges[:-1] # for x error
xsRight = xsEdges[1:] - xs # actually same as xsLeft



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
SIGMA   = ys
XERR    = xsLeft
XEDGES  = xsEdges

with open('{0}{1}.txt'.format(savefolder+'cl_values/','SIGMA_F'), 'w+') as fsigf:
    np.savetxt(fsigf, SIGMA_F)

with open('{0}{1}.txt'.format(savefolder+'cl_values/','SIGMA'), 'w+') as fsig:
    np.savetxt(fsig, SIGMA)

with open('{0}{1}.txt'.format(savefolder+'cl_values/','xs'), 'w+') as fxs:
    np.savetxt(fxs, xs)

with open('{0}{1}.txt'.format(savefolder+'cl_values/','XERR'), 'w+') as fxerr:
    np.savetxt(fxerr, XERR)

with open('{0}{1}.txt'.format(savefolder+'cl_values/','XEDGES'), 'w+') as fxed:
    np.savetxt(fxed, XEDGES)




#'''
                                #----------PLOT WITH ERROR---------# 
mpl.rcParams.update({'font.family':'serif'})
mpl.rcParams.update({'mathtext.fontset':'cm'})

color = cm.hsv(np.linspace(0, 1.1, len(data_key)+4))
plt.clf()
plt.figure(7, figsize=(10,6))
for i, datakey in enumerate(OWLS_datakey): 
    plt.semilogx(lb, clBary[i], color=color[i], label=datakey)   # (99991,) --> bin 
    #plt.errorbar(xs, BARY[i](xs), yerr=SIGMA, ecolor=colors[i], capsize=4, fmt='none')            # (15,) --> bin

i+=1
plt.semilogx(lb, cl_Hz_list[1], color=color[i], ls='--', label='Hz-DM')
i+=1
plt.semilogx(lb, cl_Hz_list[0], color=color[i], ls='--', label='Hz-AGN maybe')
i+=1
plt.semilogx(lb, cl_Hz_nofix_list[0], color=color[i], ls='--', label='Hz-AGN nofix')
i+=1
plt.semilogx(lb, cl_Hz_fix_list[0], color=color[i], ls='--', label='Hz-AGN fix')


plt.semilogy(l, cl_camb, color='k', label='cl CAMB no baryon')
plt.title(r'$C_\ell^{\kappa\kappa}$ BARYON', size=20)
plt.ylabel(r'$C_\ell^{\kappa\kappa, bary}$', size=20)
plt.xlabel(r'$\ell$', size=20)
plt.xlim(5000, 1e5)
plt.ylim(2e-12,1e-9)
plt.grid(True)
plt.legend(ncol=2, loc='upper right', prop={'size': 9.5})
#plt.show()
plt.savefig('{0}cl_plots/cl_OWLS_Hz.pdf'.format(savefolder))

# ------------------------------------------------------------------------

# Cl ratio plot with zoom in
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
from matplotlib.patches import Patch
alpha = 0.3

def make_error_boxes(ax, xdata, ydata, xerror, yerror, facecolor='red', edgecolor='None', alpha=alpha):
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

color = cm.hsv(np.linspace(0, 1.2, len(data_key)+4))

plt.clf()
fig, ax = plt.subplots(1, figsize=(11,8))
for i, datakey in enumerate(OWLS_datakey): 
    label=datakey.replace('_L100N512','')
    if i == base_index:
        label='DMONLY (DMO)'
    ax.semilogx(lb, clBary[i]/clBary[base_index], color=color[i], label=label)   # (99991,) --> bin 

i+=1
ax.semilogx(lb, cl_Hz_list[0]/cl_Hz_list[1], color=color[i], label='Hz-AGN maybe', ls='--')
i+=1
ax.semilogx(lb, cl_Hz_nofix_list[0]/cl_Hz_nofix_list[1], color=color[i], label='Hz-AGN nofix', ls='--')
i+=1
ax.semilogx(lb, cl_Hz_fix_list[0]/cl_Hz_fix_list[1], color=color[i], label='Hz-AGN fix', ls='--')

leg = ax.legend(ncol=3, loc='upper left', prop={'size': 11})

axins = ax.inset_axes([0.09, 0.25, 0.6, 0.52])
for i, datakey in enumerate(OWLS_datakey): 
    axins.plot(lb, clBary[i]/clBary[base_index], color=colors[i], label=datakey) 

i+=1
axins.semilogx(lb, cl_Hz_list[0]/cl_Hz_list[1], color=color[i], label='Hz-AGN maybe', ls='--')
i+=1
axins.semilogx(lb, cl_Hz_nofix_list[0]/cl_Hz_nofix_list[1], color=color[i], label='Hz-AGN nofix', ls='--')
i+=1
axins.semilogx(lb, cl_Hz_fix_list[0]/cl_Hz_fix_list[1], color=color[i], label='Hz-AGN fix', ls='--')

xdata = xs
ydata = np.ones(xs.shape)       # all ones because Cl_DMO/Cl_DMO
xerr  = np.array([xsRight,xsLeft])
yerr  = SIGMA_F
yerr  = np.array([yerr,yerr])

artist = make_error_boxes(axins, xdata, ydata, xerr, yerr, facecolor='darkviolet')
legend_elements = [Patch(facecolor='darkviolet', edgecolor='None', label='CMB-HD', alpha=alpha)]
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


# ------------------------------------------------------------------------
plt.clf()
plt.figure(7, figsize=(10,6))
zlist = Hz_data['Hz-AGN']['z']
k = Hz_data['Hz-AGN']['k']
color = cm.hsv(np.linspace(0, 0.9, len(zlist)))


for i, z in enumerate(zlist[:5]):
    plt.semilogx(k, Hz_data_fix['Hz-AGN']['R_interpolator'](k,z) , color=colors[i], label='z = ' + str(np.around(z, decimals=2)) + ' fix')
    plt.semilogx(k, Hz_data['Hz-AGN']['R_interpolator'](k,z) , color=colors[i+5], ls='-.',label='z = ' + str(np.around(z, decimals=2)) + ' maybe fix')
    plt.semilogx(k, Hz_data_nofix['Hz-AGN']['R_interpolator'](k,z) , color=color[i], ls='dotted',label='z = ' + str(np.around(z, decimals=2)) + ' nofix')

plt.title('Horizon Ratio', size=20)
plt.ylabel(r'$P_{AGN}$/$P_{DM}$', size=20)
plt.xlabel(r'$k$ [$h/Mpc$]', size=20)
plt.legend(ncol=3)
plt.xlim(k[0], 1.4)
plt.ylim(0.970, 1.025)
plt.grid(True)
#plt.show()
plt.savefig('{0}cl_plots/pk_Hz.pdf'.format(savefolder))

#'''
ending = time.time()
print('Everything took: ', ending - beginning)
