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
from header.import_baryon_data import import_data, interpolate_ratio, interpolate_ratio_Hz
import matplotlib.cm as cm
import copy


savefolder = 'outputs_jul17/'
datafolder = 'baryonic_data'

#color = cm.hsv(np.linspace(0, 1.7, len(xs)))
colors = np.array(['r', 'darkorange', 'gold', 'limegreen', 'forestgreen','deepskyblue', 'blue','darkviolet', 'fuchsia','brown'])

######################################################################################################### 
#                                     IMPORTING BARYONIC DATA                                           #
######################################################################################################### 
data_key, data  = import_data()
OWLS_datakey    = []
BAHAMAS_datakey = []
Hz_datakey      = []
for i, key in enumerate(data_key):
    if 'Hz' in key:
        Hz_datakey.append(key)
    elif 'BAHAMAS' in key:
        BAHAMAS_datakey.append(key)
    else:
        OWLS_datakey.append(key)

# These are the "data_same" stuff with matching z's
# ------------ OWLS stuff ---------------
OWLS_basekey   = 'DMONLY_L100N512'
OWLS_datakey   = np.array(OWLS_datakey)
OWLS_baseindex_all = int(np.argwhere(data_key==OWLS_basekey)) # from all sims
OWLS_baseindex = int(np.argwhere(OWLS_datakey==OWLS_basekey)) # OWLS
OWLS_data      = interpolate_ratio(OWLS_datakey, data, OWLS_baseindex) #for OWLS only

data_same       = OWLS_data         # from old code
base_index      = OWLS_baseindex   # from old code

# ------------ OWLS stuff ---------------
BAHAMAS_basekey   = 'BAHAMAS-DMONLY'
BAHAMAS_datakey   = np.array(BAHAMAS_datakey)
BAHAMAS_baseindex_all = int(np.argwhere(data_key==BAHAMAS_basekey)) # from all sims
BAHAMAS_baseindex = int(np.argwhere(BAHAMAS_datakey==BAHAMAS_basekey)) # BAHAMAS
BAHAMAS_data      = interpolate_ratio(BAHAMAS_datakey, data, BAHAMAS_baseindex) #for OWLS only

# Mismatched z's
# ------------ Horizon stuff ---------------
Hz_basekey   = 'Hz-DM'
Hz_baseindex_all = int(np.argwhere(data_key==Hz_basekey)) # from all sims
Hz_baseindex = int(np.argwhere(data_key==Hz_basekey))   # Horizon
Hz_datakey   = np.array(Hz_datakey)

Hz_data, Hz_data_fix, Hz_data_nofix = copy.deepcopy(data),copy.deepcopy(data),copy.deepcopy(data)
Hz_data_nofix = interpolate_ratio_Hz(Hz_datakey, Hz_data_nofix, fix=False)
#Hz_data       = interpolate_ratio_Hz(data_key, Hz_data, fix='Maybe')
#Hz_data_fix   = interpolate_ratio_Hz(data_key, Hz_data_fix, fix=True)

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
cl_OWLS_list     = GetBaryonLensingPower(OWLS_datakey, OWLS_data, lmax, dl, nz, which_sim='OWLS', save=True, savefolder=savefolder+'cl_values/', savename='cl_OWLS_list_lmax1e5')
cl_BAHAMAS_list  = GetBaryonLensingPower(BAHAMAS_datakey, BAHAMAS_data, lmax, dl, nz, which_sim='BAHAMAS', save=True, savefolder=savefolder+'cl_values/', savename='cl_BAHAMAS_list_lmax1e5')

#cl_Hz_list       = GetBaryonLensingPower(Hz_datakey, Hz_data, lmax, dl, nz, which_sim='Hz', save=True, savefolder=savefolder+'cl_values/', savename='cl_Hz_list_lmax1e5')
cl_Hz_nofix_list = GetBaryonLensingPower(Hz_datakey, Hz_data_nofix, lmax, dl, nz, which_sim='Hz', save=True, savefolder=savefolder+'cl_values/', savename='cl_Hz_nofix_list_lmax1e5')
#cl_Hz_fix_list   = GetBaryonLensingPower(Hz_datakey, Hz_data_fix, lmax, dl, nz, which_sim='Hz', save=True, savefolder=savefolder+'cl_values/', savename='cl_Hz_fix_list_lmax1e5')

'''

######################################################################################################### 
#                                         IMPORT CL VALUES                                              #
######################################################################################################### 
                                #-------------CAMB CL-------------# 
cl_camb   = np.array(np.loadtxt('{0}cl_values/cl_camb.txt'.format(savefolder)))    # l from 0 to 40000
#cl_weyl  = np.loadtxt('{0}cl_values/cl_weyl.txt'.format(savefolder))    # l from 0 to 5000
#cl_delta = np.loadtxt('{0}cl_values/cl_delta.txt'.format(savefolder))   # l from 0 to 5000
l         = np.array(np.loadtxt('{0}cl_values/l.txt'.format(savefolder))) # currently range from 0 to 40000

                                #-----------BARYONIC CL-----------# 
#cl_OWLS_list    = np.array(np.loadtxt('{0}cl_values/cl_OWLS_list.txt'.format(savefolder)))
cl_OWLS_list     = np.array(np.loadtxt('{0}cl_values/cl_OWLS_list_lmax1e5.txt'.format(savefolder)))
cl_BAHAMAS_list  = np.array(np.loadtxt('{0}cl_values/cl_BAHAMAS_list_lmax1e5.txt'.format(savefolder)))

#cl_Hz_list      = np.array(np.loadtxt('{0}cl_values/cl_Hz_list_lmax1e5.txt'.format(savefolder)))
cl_Hz_nofix_list = np.array(np.loadtxt('{0}cl_values/cl_Hz_nofix_list_lmax1e5.txt'.format(savefolder)))
#cl_Hz_fix_list  = np.array(np.loadtxt('{0}cl_values/cl_Hz_fix_list_lmax1e5.txt'.format(savefolder)))

lb               = np.array(np.loadtxt('{0}cl_values/lb.txt'.format(savefolder)))


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

cl_allsim_list = np.concatenate((cl_OWLS_list, cl_BAHAMAS_list, cl_Hz_nofix_list))
# order according to data_key
   
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
''' MUST CHANGE CODE TO REFLECT NEW SIMS
BARY = [] # interpolated cl for all bary
for clbary in cl_allsim_list:
    BARY.append(interpolate.interp1d(lb,clbary,bounds_error=True))
                                    
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
'''
DMO     = interpolate.interp1d(lb,cl_allsim_list[OWLS_baseindex],bounds_error=True)
SIGMA_F = ys/DMO(xs)  # (15, ) --> bin # fractional sigma
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

color = cm.hsv(np.linspace(0, 0.9, len(data_key)))
plt.clf()
plt.figure(7, figsize=(10,6))
for i, datakey in enumerate(data_key): 
    plt.semilogx(lb, cl_allsim_list[i], color=color[i], label=datakey)   # (99991,) --> bin 
    #plt.errorbar(xs, BARY[i](xs), yerr=SIGMA, ecolor=colors[i], capsize=4, fmt='none')            # (15,) --> bin

plt.semilogy(l, cl_camb, color='k', label='cl CAMB no baryon')
plt.title(r'$C_\ell^{\kappa\kappa}$ BARYON', size=20)
plt.ylabel(r'$C_\ell^{\kappa\kappa, bary}$', size=20)
plt.xlabel(r'$\ell$', size=20)
plt.xlim(5000, 1e5)
plt.ylim(2e-12,1e-9)
plt.grid(True)
plt.legend(ncol=3, loc='upper right', prop={'size': 9.5})
#plt.show()
plt.savefig('{0}cl_plots/cl_allsim.pdf'.format(savefolder))

# ------------------------------------------------------------------------

# Cl ratio plot with zoom in
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
from matplotlib.patches import Patch
alpha = 0.4

datakey_list = np.copy(data_key)
for i, key in enumerate(datakey_list):
    if 'Hz' in key:
        datakey_list[i] = key.replace('Hz','Horizon')
    elif 'BAHAMAS' in key:
        pass
    else:
        datakey_list[i] = 'OWLS-' + key.replace('_L100N512','')

def make_error_boxes(ax, xdata, ydata, xerror, yerror, facecolor='gray', edgecolor='None', alpha=alpha):
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


ind = [7,0,1,14,11,10,12] #[0,1,2,4,7,10,11,12,14] AGN, NOSN, REF, WDENS, DMONLY, BAHAMAS-AGN, -LowAGN, -HighAGN, Hz-AGN
plot_key = data_key[ind]
plot_cl  = cl_allsim_list[ind]
OWLS_DMO = cl_allsim_list[OWLS_baseindex_all]
BAHAMAS_DMO = cl_allsim_list[BAHAMAS_baseindex_all]
Hz_DMO = cl_allsim_list[Hz_baseindex_all]

cold = np.flip(cm.jet(np.linspace(0.07, 1.0, 9)), axis=0) 
#rainbow = cm.hsv(np.linspace(0, 0.9, len(ind)))


fig, ax = plt.subplots(1, figsize=(11,6.5))
lines, labels = [],[]

for i, datakey in enumerate(plot_key):  
    label = datakey
    
    if 'Hz' in datakey:
        line, = ax.semilogx(lb, plot_cl[i]/Hz_DMO, color=cold[i], label=label)
    elif 'BAHAMAS' in datakey:
        line, = ax.semilogx(lb, plot_cl[i]/BAHAMAS_DMO, color=cold[i+2], label=label)
    else:
        color = cold[i]
        label = 'OWLS-' + datakey.replace('_L100N512', '')
        if i == 0:
            label='DMONLY (DMO)'
            color='k'
        line, = ax.semilogx(lb, plot_cl[i]/OWLS_DMO, color=color, label=label)
    
    lines.append(line)
    labels.append(label)


# LABEL: swap the positions of Hz-HighAGN and Hz-LowAGN
lines, labels               = np.array(lines), np.array(labels)
lines[-1],  lines[-3]  = lines[-3],  lines[-1]
labels[-1], labels[-3] = labels[-3], labels[-1]

leg = ax.legend(lines[1:], labels[1:], ncol=2, prop={'size': 12}, bbox_to_anchor=(0.33, 0.5, 0.3, 0.5))

# Zoomed plot
axins = ax.inset_axes([0.07, 0.24, 0.6, 0.57])
for i, datakey in enumerate(plot_key): 
    label = datakey
    
    if 'Hz' in datakey:
        axins.plot(lb, plot_cl[i]/Hz_DMO, color=cold[i])
    elif 'BAHAMAS' in datakey:
        axins.plot(lb, plot_cl[i]/BAHAMAS_DMO, color=cold[i+2])
    else:
        color = cold[i]
        if i == 0:
            label='DMONLY (DMO)'
            color='k'
        axins.plot(lb, plot_cl[i]/OWLS_DMO, color=color, label=label)

xdata = xs
ydata = np.ones(xs.shape)       # all ones because Cl_DMO/Cl_DMO
xerr  = np.array([xsRight,xsLeft])
yerr  = SIGMA_F
yerr  = np.array([yerr,yerr])

artist = make_error_boxes(axins, xdata, ydata, xerr, yerr, facecolor='darkviolet')
lines[0].set_label(labels[0])
legend_elements = [Patch(facecolor='darkviolet', edgecolor='None', label='CMB-HD', alpha=alpha), lines[0]]
legin = ax.legend([artist],handles=legend_elements,loc='upper right', prop={'size': 12}, bbox_to_anchor=(0.63, 0.7, 0.3, 0.3))
ax.add_artist(leg)
ax.add_artist(legin)

axins.set_xlim(10000, 37500)
axins.set_ylim(0.78, 1.22)
axins.grid(True)
axins.tick_params(direction='inout', grid_alpha=0.5, labelsize=13)

ax.indicate_inset_zoom(axins)
ax.tick_params(direction='inout', grid_alpha=0.5, labelsize=14, length=7)

mark_inset(ax, axins, loc1=1, loc2=3, fc='none', ec='0.5')

plt.title('Ratio of Baryonic and Dark Matter-Only '+r'Lensing Power Spectra $C_\ell^{\kappa\kappa}$', size=18)
plt.ylabel(r'$C_\ell^{\kappa\kappa, bary}$/$C_\ell^{\kappa\kappa, DMO}$', size=20)
plt.xlabel(r'$\ell$', size=25)
plt.ylim(0.76)
#plt.show()
filename = savefolder + 'cl_plots/cl_ratio_zoomin_error_poster.pdf'
plt.savefig(filename, bbox_inches='tight', pad_inches=0.1)


# ------------------------------------------------------------------------
# CHANGE CODE FOR DIFFERENT VARIABLE NAMES
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

# ------------------------------------------------------------------------
plt.clf()
plt.figure(7, figsize=(10,6))
plt.semilogx(lb, cl_Hz_fix_list[0]/cl_Hz_nofix_list[0], color='r', label='Hz-AGN fix/nofix')
plt.semilogx(lb, cl_Hz_fix_list[1]/cl_Hz_nofix_list[1], color='k', label='Hz-DM fix/nofix')
plt.title(r'Horizon $C_\ell^{\kappa\kappa}$ Fix Ratio', size=20)
plt.ylabel(r'$C_\ell^{\kappa\kappa, fix}$/$C_\ell^{\kappa\kappa, nofix}$', size=20)
plt.xlabel(r'$\ell$', size=20)
#plt.legend()
#plt.xlim(k[0], 1.4)
#plt.ylim(0.970, 1.025)
plt.grid(True)
#plt.show()
plt.savefig('{0}cl_plots/cl_Hz_fix_nofix.pdf'.format(savefolder))

#'''
ending = time.time()
print('Everything took: ', ending - beginning)
