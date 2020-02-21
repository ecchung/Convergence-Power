# FOR BARYONIC PAPER REVIEW REVISION FEB 20, 2020
# Taken from mainscript.py
# Last edited Feb 20, 2020

import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from header.CAMB_header import *
from header.import_baryon_data import import_data, interpolate_ratio
import copy


importfolder='outputs_jul17/'
savefolder = '../../Fisher/pyfisher/myoutput/plots/'
datafolder = 'baryonic_data'

######################################################################################################### 
#                                     IMPORTING BARYONIC DATA                                           #
######################################################################################################### 
data_key, data  = import_data()
OWLS_datakey    = []
BAHAMAS_datakey = []
Hz_datakey      = []
TNG100_datakey  = []
TNG300_datakey  = []

for i, key in enumerate(data_key):
    if 'Hz' in key:
        Hz_datakey.append(key)
    elif 'BAHAMAS' in key:
        BAHAMAS_datakey.append(key)
    elif 'OWLS' in key:
        OWLS_datakey.append(key)
    elif 'TNG100' in key:
        TNG100_datakey.append(key)
    elif 'TNG300' in key:
        TNG300_datakey.append(key) 

print('Do dict.keys() to find the appropriate keys.')

# These are the "data_same" stuff with matching z's
# ------------ OWLS stuff ---------------
OWLS_basekey   = 'OWLS-DMONLY'
OWLS_datakey   = np.array(OWLS_datakey)
OWLS_baseindex_all = int(np.argwhere(data_key==OWLS_basekey)) # from all sims
OWLS_baseindex = int(np.argwhere(OWLS_datakey==OWLS_basekey)) # OWLS
OWLS_data      = interpolate_ratio(OWLS_datakey, data, 'OWLS', sim_baseindex=OWLS_baseindex) #for OWLS only

data_same      = OWLS_data        # from old code
base_index     = OWLS_baseindex   # from old code

# ------------ BAHAMAS stuff ---------------
BAHAMAS_basekey   = 'BAHAMAS-DMONLY'
BAHAMAS_datakey   = np.array(BAHAMAS_datakey)
BAHAMAS_baseindex_all = int(np.argwhere(data_key==BAHAMAS_basekey)) # from all sims
BAHAMAS_baseindex = int(np.argwhere(BAHAMAS_datakey==BAHAMAS_basekey)) # BAHAMAS
BAHAMAS_data      = interpolate_ratio(BAHAMAS_datakey, data, 'BAHAMAS', sim_baseindex=BAHAMAS_baseindex) 

# Mismatched z's
# ------------ Horizon stuff ---------------
Hz_basekey   = 'Hz-DM'
Hz_baseindex_all = int(np.argwhere(data_key==Hz_basekey)) # from all sims
Hz_baseindex = int(np.argwhere(data_key==Hz_basekey))   # Horizon
Hz_datakey   = np.array(Hz_datakey)

Hz_data, Hz_data_fix, Hz_data_nofix = copy.deepcopy(data),copy.deepcopy(data),copy.deepcopy(data)
Hz_data_nofix = interpolate_ratio(Hz_datakey, Hz_data_nofix, 'Hz', fix=False)
#Hz_data       = interpolate_ratio_Hz(data_key, Hz_data, fix='Maybe')
#Hz_data_fix   = interpolate_ratio_Hz(data_key, Hz_data_fix, fix=True)

# ------------ Illustris TNG stuff ---------------
TNG100_basekey   = 'TNG100DM'
TNG100_datakey   = np.array(TNG100_datakey)
TNG100_baseindex_all = int(np.argwhere(data_key==TNG100_basekey)) # from all sims
TNG100_baseindex = int(np.argwhere(TNG100_datakey==TNG100_basekey)) # TNG100
TNG100_data      = interpolate_ratio(TNG100_datakey, data, 'TNG100') 

TNG300_basekey   = 'TNG300DM'
TNG300_datakey   = np.array(TNG300_datakey)
TNG300_baseindex_all = int(np.argwhere(data_key==TNG300_basekey)) # from all sims
TNG300_baseindex = int(np.argwhere(TNG300_datakey==TNG300_basekey)) # TNG300
TNG300_data      = interpolate_ratio(TNG300_datakey, data, 'TNG300') 

######################################################################################################### 
#                                   GETTING THE CLKK INTEGRAND                                          #
######################################################################################################### 

which_simset = 'BAHAMAS'
which_sim    = 'BAHAMAS-HighAGN'

if (which_simset == 'BAHAMAS'):
    dataset = BAHAMAS_data
    sim_DMO = BAHAMAS_basekey
    kmax    = 514.71854
    kmin    = 0.015707963
# add more elifs as needed...

ell = 1000 # fix ell

# original data:
X  = dataset[which_sim]['X'][1:] # first element 0; k_l gets divide by zero
z  = dataset[which_sim]['z'][1:]
a  = get_a(z)
Wk = W(X, a, Xcmb)
kl = (ell + 0.5)/X  # k at the given ell for each X

# baryonic ones
P_int_bary = dataset[which_sim]['P_interpolator'] # P_int(k, z); k and/or z can be a single value or a list

Pl_bary    = np.diag(np.flip(P_int_bary(kl, z), axis=1))  
# bad habit of resorting k's from lowest to highest, so I flip them
# also takes each z for each k_l, but I only want pairs of k_l[i], z[i]
I_bary     = (Wk/X)**2 * Pl_bary # Clkk integrand

# DMO ones
P_int_DMO = dataset[sim_DMO]['P_interpolator'] 

Pl_DMO    = np.diag(np.flip(P_int_DMO(kl, z), axis=1))
I_DMO     = (Wk/X)**2 * Pl_DMO 



# interpolated data
n   = 500
Xs  = np.linspace(0, Xcmb, n)[1:]
zs  = results.redshift_at_comoving_radial_distance(Xs)
As  = get_a(zs)
Ws  = W(Xs, As, Xcmb)
kls = (ell + 0.5)/Xs

# baryonic ones
Pls_bary    = np.diag(np.flip(P_int_bary(kls, zs), axis=1))  
Is_bary     = (Ws/Xs)**2 * Pls_bary 

# DMO ones
Pls_DMO    = np.diag(np.flip(P_int_DMO(kls, zs), axis=1))
Is_DMO     = (Ws/Xs)**2 * Pls_DMO 



# plot
fig, ax= plt.subplots(1, figsize=(8,6))
ax.semilogx(zs[Xs>400],Pls_bary[Xs>400]/Pls_DMO[Xs>400], label='interpolated')
ax.semilogx(z, Pl_bary/Pl_DMO,c='red',ls='None', marker='.',label='raw data')
plt.ylabel(r'$\frac{\hat P_{\rm bary}(k,z)}{\hat P_{\rm DMO}(k,z)}$')
plt.xlabel(r'$z$')
plt.title(r'BAHAMAS-HighAGN at $\ell=1000$')
plt.legend()
#plt.show()
plt.savefig(savefolder + 'feb20_2020_P_ratio_l=1000_BAHAMAS_HighAGN.pdf')



