# BARYONIC ANALYSIS
# Last edited June 25, 2019

import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
import camb
from camb import model, initialpower
from scipy.constants import c           # c / 1000 = 299 km/s
import scipy.integrate as integrate
from scipy import interpolate
import time

savefolder   = 'jul08_baryon_analysis/'
datafolder   = 'baryonic_data'

beginning = time.time()

# Define parameters used
H0    = 67.5            # Hubble constant today         (km/s/Mpc)
H0c   = H0/c * 1000     # Hubble constant today         (1/Mpc)
h     = H0/100          # H0 / 100 km/s /Mpc            (unitless)
ombh2 = 0.022           # baryonic matter density       (unitless)
omch2 = 0.12            # cold dark matter density      (unitless)
omb   = ombh2 / h**2    # baryonic matter density fraction  (unitless)
omc   = omch2 / h**2    # cold dark matter density fraction (unitless)
omm   = omb + omc       # total matter density          (unitless)
oml   = 1 - (omb + omc) # omega lambda                  (unitless)
omk   = 0.0             # curvature parameter           (unitless)
ns    = 0.965           # scalar spectral index
lmax  = 5000
l     = np.arange(1,lmax+1)
ls    = np.arange(2, lmax+1, dtype=np.float64)
lim   = l + 0.5 # limber approximation l


# Set initial paramters using CAMB
pars    = camb.CAMBparams()
pars.set_cosmology(H0 = H0, ombh2=ombh2, omch2=omch2) 
pars.InitPower.set_params(ns = ns)
results = camb.get_background(pars)
Xcmb    = results.conformal_time(0)- results.tau_maxvis
zcmb    = results.redshift_at_comoving_radial_distance(Xcmb)
z       = np.logspace(-5.,np.log10(zcmb), 1500, endpoint=True) 
maxkh   = 2000.
npoints = 1000
lpa     = 4  # lens_potential_accuracy


########################################################################################################
#                                         FUNCTION DEFINITIONS                                         #
########################################################################################################

# Redshift
def get_z(a):
    return 1/a - 1
    
# Scaling factor 
def get_a(z):
    return 1. / (1. + z)

# Comoving distance X(z)
def get_chi(results, z):
    return results.comoving_radial_distance(z)

# Wavenumber k(l, X)
def get_k(l, chi):
    return l/chi
    
# Hubble Parameter H(z)
def H(z):
    '''
    Returns in 1/Mpc.
    '''
    return np.vectorize(results.h_of_z)(z)
    
# Omega_m(z)
def Omm(z):
    return omm * H0c**2 * (1+z) / H(z)**2

# Weight function W(X)
def W(chi, a, xcmb): # call a meshgrid of chi and a
    return (3/2) * omm * H0c**2 * chi/a * (1-chi/xcmb)

# Matter Power Spectrum    
def P_delta(z, k, from_func='Weyl'):
    ''' 
    from_func can take 'Weyl', 'psi', 'Weyl linear'
    Defaults to from_func = 'Weyl'
    '''
    if from_func == 'Weyl':
        return 4 / (9 * Omm(z)**2 * H(z)**4) * PK_weyl.P(z, k, grid=False)
    elif from_func == 'psi':
        return 4 / (9 * Omm(z)**2 * H(z)**4 * k**3) * PK_weyl.P(z, k, grid=False)
    if from_func == 'Weyl linear':
        return 4 / (9 * Omm(z)**2 * H(z)**4) * PK_weyl_linear.P(z, k, grid=False)
    else:
        print('Invalid argument for from_func')
   
  
########################################################################################################
#                                        GET P(K,Z) IN L-X SPACE                                       #
########################################################################################################
print('Getting matter power interpolators')

#PK_tot_k = camb.get_matter_power_interpolator(pars, nonlinear=True, hubble_units=False, k_hunit=False, kmax=maxkh, var1=model.Transfer_tot, var2=model.Transfer_tot, zmax=zcmb)

PK_weyl  = camb.get_matter_power_interpolator(pars, nonlinear=True, hubble_units=False, k_hunit=False, kmax=maxkh, var1=model.Transfer_Weyl, var2=model.Transfer_Weyl, zmax=zcmb)

#PK_weyl_linear  = camb.get_matter_power_interpolator(pars, nonlinear=False, hubble_units=False, k_hunit=False, kmax=maxkh, var1=model.Transfer_Weyl, var2=model.Transfer_Weyl, zmax=zcmb)


######################################################################################################### 
#                                     AUTOMATIC CAMB CL CALCULATION                                     #
#########################################################################################################
'''
# CAMB output
print('Calculating Cl automatically')
pars.set_for_lmax(lmax, lens_potential_accuracy=lpa)
results = camb.get_results(pars)
cl_camb = results.get_lens_potential_cls(lmax) 
cl_camb = cl_camb * 2 * np.pi / 4 
cl_camb = cl_camb[:,0][1:]


######################################################################################################### 
#                                       MANUAL CAMB CL CALCULATION                                      #
#########################################################################################################
# Cl CAMB manual original using P_weyl
print('Calculating Cl manual original')
nz  = 100
Xs  = np.linspace(0, Xcmb, nz)
zs  = results.redshift_at_comoving_radial_distance(Xs)

dXs = (Xs[2:]-Xs[:-2])/2
Xs  = Xs[1:-1]
zs  = zs[1:-1]

# Get lensing window function (flat universe)
ws  = ((Xcmb - Xs)/(Xs**2 * Xcmb))**2
ls  = np.arange(2, lmax+1, dtype=np.float64)
d   = np.ones(Xs.shape)
cl_kappa_orig = np.zeros(ls.shape)

for i, li in enumerate(ls):
    k = (li+0.5) / Xs   #kkhc
    d[:] = 1
    d[k<1e-4] = 0
    d[k>=maxkh] = 0
    cl_kappa_orig[i] = np.dot(dXs, d*PK_weyl.P(zs, k, grid=False)*ws / k**4)

cl_kappa_orig = cl_kappa_orig * (ls*(ls+1))**2

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Cl P_weyl -> P_psi : works well!!!!!!
print('Calculating Cl with input P_weyl')

nz  = 100
Xs  = np.linspace(0, Xcmb, nz)
zs  = results.redshift_at_comoving_radial_distance(Xs)

dXs = (Xs[2:]-Xs[:-2])/2
Xs  = Xs[1:-1]
zs  = zs[1:-1]
As  = get_a(zs)

# Get lensing window function (flat universe)
ws = W(Xs, As, Xcmb)
ls = np.arange(2, lmax+1, dtype=np.float64)
d  = np.ones(Xs.shape)
cl_kappa_dweyl = np.zeros(ls.shape)

for i, li in enumerate(ls):
    k = (li+0.5) / Xs
    d[:] = 1
    d[k<1e-4] = 0
    d[k>=maxkh] = 0
    cl_kappa_dweyl[i] = np.dot(dXs, d * P_delta(zs, k, from_func='Weyl') * ws**2 / Xs**2)
    
    
######################################################################################################### 
#                                             SAVING CL VALUES                                          #
#########################################################################################################  
with open('{0}cl_values/cl_camb.txt'.format(savefolder), 'w+') as fcl:
    np.savetxt(fcl, cl_camb)
with open('{0}cl_values/cl_kappa_orig.txt'.format(savefolder), 'w+') as fcl_orig:
    np.savetxt(fcl_orig, cl_kappa_orig)
with open('{0}cl_values/cl_kappa_dweyl.txt'.format(savefolder), 'w+') as fcl_dweyl:
    np.savetxt(fcl_dweyl, cl_kappa_dweyl)
      
'''
######################################################################################################### 
#                                             IMPORT CL VALUES                                          #
######################################################################################################### 
cl_camb        = np.loadtxt('{0}cl_values/cl_camb.txt'.format(savefolder))
cl_kappa_orig  = np.loadtxt('{0}cl_values/cl_kappa_orig.txt'.format(savefolder))
cl_kappa_dweyl = np.loadtxt('{0}cl_values/cl_kappa_dweyl.txt'.format(savefolder))


#########################################################################################################
#                                                   PLOTS                                               #
#########################################################################################################
print('Plotting')
'''
# Plot my Cl and camb's Cl for nonlinear growth
plt.figure(6, figsize=(10,6))
plt.clf()
plt.loglog(l[1:] , cl_camb[1:], color='g', label=r'$C_\ell^{\kappa\kappa}$ CAMB Function')
plt.loglog(ls, cl_kappa_orig, color='b', label=r'$C_\ell^{\kappa\kappa}$ CAMB manual original ')
plt.loglog(ls, cl_kappa_dweyl, color='magenta', label=r'$C_\ell^{\kappa\kappa}$ manual P_weyl(P_delta)')

plt.title(r'$C_\ell^{\kappa\kappa}$ Nonlinear')
plt.ylabel(r'Convergence Power $C_\ell^{\kappa\kappa}$')
plt.xlabel(r'Multipole $\ell$')
plt.grid(True)
plt.legend()
plt.savefig('{0}cl_plots/cl_compare.pdf'.format(savefolder))

# ---------------------------------------------------------------------------------------------------------

cl_camb_interp = interpolate.interp1d(l, cl_camb)
cl_camb_int = cl_camb_interp(ls)

# Plot ratio of my Cl and camb's Cl NONLINEAR
plt.figure(7, figsize=(10,6))
plt.clf()

plt.plot(ls[10:], (cl_kappa_orig[10:]-cl_camb_int[10:])/cl_camb_int[10:]*100, color='b', label=r'$\frac{C_\ell^{\kappa\kappa} (CAMB original) - C_\ell^{\kappa\kappa} (camb interp)}{C_\ell^{\kappa\kappa} (camb interp)}$')
plt.plot(ls[10:], (cl_kappa_dweyl[10:]-cl_camb_int[10:])/cl_camb_int[10:]*100, color='magenta', label=r'$\frac{C_\ell^{\kappa\kappa} (dweyl) - C_\ell^{\kappa\kappa} (camb interp)}{C_\ell^{\kappa\kappa} (camb interp)}$')

plt.title(r'Percentage Difference $C_\ell$ mine vs. $C_\ell$ camb')
plt.ylabel(r'Convergence Power $C_\ell^{\kappa\kappa}$ Difference (%)')
plt.xlabel(r'Multipole $\ell$')
plt.grid(True)
plt.ylim(-0.4)
plt.legend(loc='center right', bbox_to_anchor=(0.5, 0.4), fancybox=True)
plt.savefig('{0}cl_plots/cl_perdiff.pdf'.format(savefolder))
'''


######################################################################################################### 
# ----------------------------------------------------------------------------------------------------- #
######################################################################################################### 
#                                                                                                       #
#                                             BARYONIC STUFF                                            #
#                                                                                                       #
######################################################################################################### 
# ----------------------------------------------------------------------------------------------------- #
######################################################################################################### 


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
        cl_kappa_bary[i] = np.dot(dXb, d * P_delta(zb, k, from_func='Weyl') * np.diagonal((np.flip(bary_P_int(k, zb), axis=1)/np.base_P_int(k, zb))) * wb**2 / Xb**2)
        # The [:,0] is sort of sketchy 
    cl_baryon_list.append(cl_kappa_bary)

end = time.time()
print('Baryonic Cl took: ', end - start)


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
    
    
######################################################################################################### 
#                                    CALCULATING BARYONIC CL VALUES                                     #
#                                         Interpolated Ratios                                           #
######################################################################################################### 
start2 = time.time()
print('Calculating baryonic Cl 2')

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
lb = np.arange(10, lmax+1, 1, dtype=np.float64)
d  = np.ones(Xb.shape)
cl_baryon_list2 = []


# Calculate the integral
for j, datakey in enumerate(data_key):
    print(j, datakey)
    P_ratio_int = data_same[datakey]['P_ratio_interpolator']
    cl_kappa_bary2 = np.zeros(lb.shape) # for loop
        
    for i, li in enumerate(lb):
        k = (li+0.5) / Xb
        d[:] = 1
        d[k>=kmax] = 0
        
        cl_kappa_bary2[i] = np.dot(dXb, d * P_delta(zb, k, from_func='Weyl') * np.diagonal(np.flip(P_ratio_int(k, zb), axis=1)) * wb**2 / Xb**2) 
            # the interpolator weirdly sorts P values into increasing values of k
    cl_baryon_list2.append(cl_kappa_bary2)

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
    plt.semilogx(lb, (cl_baryon_list2[i]-cl_baryon_list2[base_index])/cl_baryon_list2[base_index], color=colors[i], label='{0} - 2 diag flip'.format(datakey))    


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