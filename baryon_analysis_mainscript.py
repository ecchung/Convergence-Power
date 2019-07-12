# BARYONIC ANALYSIS
# Last edited June 25, 2019

import numpy as np
import matplotlib.pyplot as plt
import camb
from camb import model, initialpower
from scipy.constants import c           # c / 1000 = 299 km/s
import scipy.integrate as integrate
from scipy import interpolate
import time
from header.CAMB_header import *
from header.CAMB_cl_calc import CAMB_auto, CAMB_Weyl, CAMB_Delta, P_delta

savefolder   = 'jul08_baryon_analysis/'
datafolder   = 'baryonic_data'

beginning = time.time()


'''
######################################################################################################### 
#                                     CALCULATE & SAVE CAMB CL VALUES                                   #
######################################################################################################### 
cl_camb  = CAMB_auto(save=True, savefolder=savefolder+'cl_values/', savename='cl_camb')
cl_weyl  = CAMB_Weyl(save=True, savefolder=savefolder+'cl_values/', savename='cl_weyl')
cl_delta = CAMB_Delta(save=True, savefolder=savefolder+'cl_values/', savename='cl_delta')
'''

######################################################################################################### 
#                                             IMPORT CL VALUES                                          #
######################################################################################################### 
cl_camb  = np.loadtxt('{0}cl_values/cl_camb.txt'.format(savefolder))
cl_weyl  = np.loadtxt('{0}cl_values/cl_weyl.txt'.format(savefolder))
cl_delta = np.loadtxt('{0}cl_values/cl_delta.txt'.format(savefolder))






#                                             BARYONIC STUFF                                            #




start = time.time()
######################################################################################################### 
#                                        IMPORTING BARYONIC DATA                                        #
#########################################################################################################
# Load data file names
datafile = np.loadtxt('{0}/data.txt'.format(datafolder), dtype=str)

# Load data into a dictionary accessible with keys that are name of the files without .txt and 'z', 'X', 'k', 'P'
data = {}
data_key = []

print('--------------------------------')
print('i datafilename')
for i, data_i in enumerate(datafile):
    print(i, data_i)
    
    # Load each data listed in the datafile
    z, k, P = np.loadtxt('{0}/{1}'.format(datafolder, data_i), unpack=True, usecols=(0,1,2), skiprows=1)
    
    zval, zi     = np.unique(z, return_inverse=True)
    kval, ki     = np.unique(k, return_inverse=True)
    Pval         = np.zeros(zval.shape + kval.shape)
    Pval[zi, ki] = P
    Xval         = get_chi(results, zval)
    
    # Save the values to the data dictionary
    key = data_i.strip('.txt')
    data_key.append(key)
    data[key] = {}
    data[key]['z'] = zval
    data[key]['X'] = Xval
    data[key]['k'] = kval
    data[key]['P'] = Pval
    data[key]['P_interpolator'] = interpolate.interp2d(kval, zval, Pval, kind='cubic')
print('--------------------------------')
 
######################################################################################################### 
#                                    CALCULATING BARYONIC CL VALUES                                     #
######################################################################################################### 
# Set base data (i.e., DMONLY data)
basefile = 'DMONLY_L100N512.txt'
base_index = int(np.argwhere(datafile==basefile))
basedata = data[data_key[base_index]]               # P_DMONLY

print('Calculating baryonic Cl')
'''
nz  = 100
Xb  = np.linspace(0, Xcmb, nz)
zb  = results.redshift_at_comoving_radial_distance(Xb)
kmax= 514.71854     # largest k value from all simulations
kmin= 0.062831853   # smallest k value from all simulations

dXb = (Xb[2:]-Xb[:-2])/2
Xb  = Xb[1:-1]
zb  = zb[1:-1]
Ab  = get_a(zb)

# Get lensing window function (flat universe)
wb = W(Xb, Ab, Xcmb)
lb = np.arange(10,  lmax+1, 1, dtype=np.float64)
d  = np.ones(Xb.shape)
cl_baryon_list = []

# Calculate the integral
for j, datakey in enumerate(data_key):
    print(j, datakey)
    bary_P_int = data[datakey]['P_interpolator']
    base_P_int = basedata['P_interpolator']
    cl_kappa_bary = np.zeros(lb.shape)
    
    for i, li in enumerate(lb):
        k = (li+0.5) / Xb
        d[:] = 1
        d[k>=kmax] = 0
        cl_kappa_bary[i] = np.dot(dXb, d * P_delta(zb, k, from_func='Weyl') * np.diagonal((np.flip(bary_P_int(k, zb), axis=1)/np.flip(base_P_int(k, zb))) * wb**2 / Xb**2))
    cl_baryon_list.append(cl_kappa_bary)

end = time.time()
print('Baryonic Cl took: ', end - start)
'''

######################################################################################################### 
#                                        INTERPOLATING PK RATIOS                                        #
######################################################################################################### 

# Getting an array of z values that all simulations have
z_same = np.zeros(10)
for i, key in enumerate(data_key):
    z_vals = data[key]['z']
    if i == 0:      # Set first z's as Z
        z_same = z_vals
    else:
        for j, zj in enumerate(z_same):
            if zj in z_vals:
                pass
            else:
                z_same = np.delete(z_same, j)

X_same = get_chi(results, z_same)
k_same = data[data_key[0]]['k']     # k values are the same for all simulation data
                
# Fill this dictionary with only the values that correspond to the same z's                
data_same = {}
for i, key in enumerate(data_key):
    data_same[key] = {}
    data_same[key]['z'] = z_same
    data_same[key]['X'] = X_same
    data_same[key]['k'] = k_same
    
    z_old = data[key]['z']      # Original z
    Pk    = data[key]['P']      # Original P(k)
    
    # Deleting elements of Pk that correspond to z values not in z_same
    z_old = data[data_key[i]]['z']
    delete_index = []
    for j, zj in enumerate(z_old):
        if zj in z_same:
            pass
        else:
            delete_index.append(j)
    print(delete_index)
    Pk = np.delete(Pk, delete_index, axis=0)
    data_same[key]['P'] = Pk
    
    data_same[key]['P_interpolator'] = interpolate.interp2d(k_same, z_same, Pk, kind='cubic')
    
# Add interpolated ratios of each simulation wrt base_data
base_same_P = data_same[data_key[base_index]]['P'] 
for i, key in enumerate(data_key):
    bary_same_P = data_same[key]['P'] 
    data_same[key]['P_ratio_interpolator'] = interpolate.interp2d(k_same, z_same, bary_same_P/base_same_P, kind='cubic', fill_value=1.)
    
'''    
######################################################################################################### 
#                                    CALCULATING BARYONIC CL VALUES                                     #
#                                         Interpolated Ratios                                           #
######################################################################################################### 
start2 = time.time()
print('Calculating baryonic Cl 2')



end2 = time.time()
print('Baryonic Cl 2 took: ', end2 - start2)     

######################################################################################################### 
#                                       SAVING CL BARYON VALUES                                         #
#########################################################################################################  
with open('{0}cl_values/cl_baryon_list_lmax={1}.txt'.format(savefolder, lmax), 'w+') as fcbl:
    np.savetxt(fcbl, cl_baryon_list)
with open('{0}cl_values/cl_baryon_list2_lmax={1}.txt'.format(savefolder, lmax), 'w+') as fcbl2:
    np.savetxt(fcbl2, cl_baryon_list2)


######################################################################################################### 
#                                        IMPORT BARYON CL VALUES                                        #
######################################################################################################### 
cl_baryon_list  = np.loadtxt('{0}cl_values/cl_baryon_list_lmax={1}.txt'.format(savefolder, 5000))
cl_baryon_list2 = np.loadtxt('{0}cl_values/cl_baryon_list2_lmax={1}.txt'.format(savefolder, 5000))


# ---------------------------------------------------------------------------------------------------------
'''
colors = np.array(['r', 'darkorange', 'yellow', 'limegreen', 'forestgreen','deepskyblue', 'blue','darkviolet', 'magenta','brown'])


'''
#########################################################################################################
#                                            BARYONIC PLOTS                                             #
#########################################################################################################
print('Plotting BARYONIC')

# Plot my Cl and camb's Cl for nonlinear growth
plt.figure(6, figsize=(10,6))
plt.clf()
plt.loglog(l , cl_camb, color='k', label='CAMB Function')
plt.loglog(ls, cl_kappa_orig, color='grey', label=r'CAMB manual original $P_{Weyl}$')
plt.loglog(ls, cl_kappa_dweyl, color='lightgray', label=r'manual $P_{\delta}(P_{Weyl})$')

for i, datakey in enumerate(data_key):
    #plt.loglog(lb, cl_baryon_list[i], color=colors[i], label='{0} - interp before ratio'.format(datakey))
    plt.loglog(lb, cl_baryon_list2[i], color=colors[i], label='{0} - 2 diag flip'.format(datakey), ls='dotted')

plt.title(r'$C_\ell^{\kappa\kappa}$ Nonlinear')
plt.ylabel(r'Convergence Power $C_\ell^{\kappa\kappa}$')
plt.xlabel(r'Multipole $\ell$')
plt.grid(True)
plt.xlim(1e2)
plt.legend(loc='center right', ncol=1, bbox_to_anchor=(1.08, 0.70), fancybox=True, prop={'size': 10})
plt.savefig('{0}cl_plots/cl_baryon.pdf'.format(savefolder))

# ---------------------------------------------------------------------------------------------------------
# Difference 

# Plot ratio of my Cl and camb's Cl NONLINEAR
plt.figure(7, figsize=(10,6))
plt.clf()

for i, datakey in enumerate(data_key):
    #plt.semilogx(np.arange(10, 5000+1, 1, dtype=np.float64), (cl_baryon_list[i]-cl_baryon_list[base_index])/cl_baryon_list[base_index], color=colors[i], label='{0} - interp before ratio'.format(datakey))    
    plt.semilogx(lb, (cl_baryon_list2[i]-cl_baryon_list2[base_index])/cl_baryon_list2[base_index], color=colors[i], label='{0}'.format(datakey))    


plt.title(r'Difference $C_\ell^{\kappa\kappa}$ DMONLY vs. $C_\ell^{\kappa\kappa}$ BARYON')
plt.ylabel(r'($C_\ell^{\kappa\kappa, bary}$ - $C_\ell^{\kappa\kappa, DMONLY}$)/$C_\ell^{\kappa\kappa, DMONLY}$')
plt.xlabel(r'Multipole $\ell$')
plt.grid(True)
plt.legend(ncol=2, loc='upper left', prop={'size': 10})
plt.savefig('{0}cl_plots/cl_diff_lmax={1}.pdf'.format(savefolder, lmax))


# ---------------------------------------------------------------------------------------------------------

plt.figure(7, figsize=(10,6))
plt.clf()

plt.semilogx(lb, cl_baryon_list[4]-cl_baryon_list[base_index], color='k', label='4 no ratio - base no ratio')    
plt.semilogx(lb, cl_baryon_list2[4]-cl_baryon_list2[base_index], color='r', label='4 ratio -  base ratio', ls='--')  

plt.title(r'Difference $C_\ell^{\kappa\kappa}$')
plt.ylabel(r'Cl')
plt.xlabel(r'Multipole $\ell$')
plt.grid(True)
plt.legend()
plt.savefig('{0}cl_plots/cl_stuff.pdf'.format(savefolder))




# ---------------------------------------------------------------------------------------------------------
# Plot P(k)'s
z0 = 3. 
k0 = np.linspace(kmin, kmax, 100000, endpoint=True)
k1 = np.logspace(-4, np.log10(kmax), 10000, endpoint=True)
#k0**3/(2*np.pi) * 

plt.figure(8, figsize=(10,6))
plt.clf()
plt.loglog(k0, P_delta(z0, k0, from_func='Weyl')*h**3, label=r'$P_{\delta}(Weyl)^{CAMB}$ Nonlinear', color='k')
#plt.loglog(k0, P_delta(z0, k0, from_func='Weyl linear')*h**3, label=r'$P_{\delta}(Weyl)^{CAMB}$ Linear', color='k')

for i, datakey in enumerate(data_key):
    plt.loglog(k0, data[datakey]['P_interpolator'](k0, z0), color=colors[i], label=datakey)
    plt.loglog(k0, data_same[datakey]['P_interpolator'](k0, z0), color=colors[i], label='{0} ratio'.format(datakey), ls='--')
    i_old = int(np.argwhere(data[datakey]['z']==z0))
    i_same = int(np.argwhere(data_same[datakey]['z']==z0))
    plt.loglog(data[datakey]['k'], data[datakey]['P'][i_old], color=colors[i], label='{0} data old'.format(datakey))
    plt.loglog(k_same, data_same[datakey]['P'][zi], color=colors[i], label='{0} data same'.format(datakey), ls='--')

plt.title('Matter Power Spectra for $z$ = {0}'.format(str(z0)))
plt.ylabel(r'$P_m(k)$ [$Mpc^3/h^3$]')
plt.xlabel(r'$k$ [$h/Mpc$]')
plt.grid(True)
plt.legend(ncol=2, prop={'size':8})
plt.savefig('{0}power_plots/powerspectra_orig_z{1}.pdf'.format(savefolder, str(int(z0))))

# ---------------------------------------------------------------------------------------------------------
# Plot P(k) ratios


i = 0
ells = np.array([100])
ell = ells[i]
k = (ell + 0.5)/Xb

bp_io = data[data_key[base_index]]['P_interpolator']
bp_is = data_same[data_key[base_index]]['P_interpolator']

p4_io = data[data_key[4]]['P_interpolator']
p4_is = data_same[data_key[4]]['P_interpolator']
pr4_is = data_same[data_key[4]]['P_ratio_interpolator']

bp_o = np.ones(98)
bp_s = np.ones(98)
p4_o = np.ones(98)
p4_s = np.ones(98)
pr4_s = np.ones(98)

for j, kj in enumerate(k):
    bp_o[j] = float(bp_io(kj, zb[j]))
    bp_s[j] = float(bp_is(kj, zb[j]))
    p4_o[j] = float(p4_io(kj, zb[j]))
    p4_s[j] = float(p4_is(kj, zb[j]))
    pr4_s[j] = float(pr4_is(kj, zb[j]))

p4_or = p4_o/bp_o
p4_sr = p4_s/bp_s

plt.figure(9, figsize=(10,6))
plt.clf()

#plt.plot(k, P_delta(zb, k4_o, from_func='Weyl')*h**3, label=r'$P_{\delta}(Weyl)$ * $h^3$', color='k')
plt.plot(k, p4_o-p4_s, color='r', label='p4_o-p4_s')
plt.plot(k, bp_o-bp_s, color='k', label='bp_o-bp_s')

plt.title(r'Power Spectra Difference for WDENS_L100N512 at $\ell=${0}'.format(str(ell)))
plt.ylabel(r'$P_{original}-P_{same}$')
plt.xlabel(r'$k$ [$h/Mpc$]')
plt.grid(True)
plt.legend()
plt.savefig('{0}power_plots/powerspectra_diff_WDENS_l={1}.pdf'.format(savefolder, str(ell)))





z0 = 0 
k0 = np.linspace(kmin, kmax, 100000, endpoint=True)
plt.figure(9, figsize=(10,6))
plt.clf()
basez0 = data[data_key[base_index]]['P_interpolator'](k0, z0)

for a, ka in enumerate(k0):
    P_ratio3[a] = P_ratio_int(ka, zb[a])

plt.semilogx(k0, np.log10(P_delta(z0, k0, from_func='Weyl')/basez0 *h**3), label=r'$P_{\delta}(Weyl)$ * $h^3$', color='k')
for i, datakey in enumerate(data_key):
    plt.semilogx(k0, np.log10(data[datakey]['P_interpolator'](k0, z0)/basez0), color=colors[i], label='{0}'.format(datakey))
    plt.semilogx(k0, np.log10(data_same[datakey]['P_ratio_interpolator'](k0, z0)), color=colors[i], label='{0} - 2'.format(datakey), ls='--')



plt.title('Power Spectra for $z$ = {0}'.format(str(z0)))
plt.ylabel(r'$log_{10}(P/P_{DMONLY})$')
plt.xlabel(r'$k$ [$h/Mpc$]')
plt.grid(True)
plt.legend(ncol=2)
plt.savefig('{0}cl_plots/powerspectra_ratio_z{1}.pdf'.format(savefolder, str(z0)))

'''

# ---------------------------------------------------------------------------------------------------------

print('savefolder: ', savefolder)

ending = time.time()
print('Everything took: ', ending - beginning)