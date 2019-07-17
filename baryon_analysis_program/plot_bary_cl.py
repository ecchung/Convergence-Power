'''
Used to plot bary Cl's.
'''
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
import sys
sys.path.append('../')
from header.CAMB_header import lmax

savefolder   = 'jul08_baryon_analysis/'

colors = np.array(['r', 'darkorange', 'yellow', 'limegreen', 'forestgreen','deepskyblue', 'blue','darkviolet', 'magenta','brown'])


######################################################################################################### 
#                                             IMPORT CL VALUES                                          #
######################################################################################################### 
cl_bary_list = np.loadtxt('{0}cl_values/cl_camb.txt'.format(savefolder))
lb           = np.loadtxt('{0}cl_values/ls.txt'.format(savefolder))
data_key     = np.loadtxt('{0}cl_values/data_key.txt'.format(savefolder))

base_key = 'DMONLY_L100N512'
base_index = int(np.argwhere(data_key==base_key))

# Import CAMB cl's
cl_camb  = np.loadtxt('{0}cl_values/cl_camb.txt'.format(savefolder))
cl_weyl  = np.loadtxt('{0}cl_values/cl_weyl.txt'.format(savefolder))
cl_delta = np.loadtxt('{0}cl_values/cl_delta.txt'.format(savefolder))
ls       = np.loadtxt('{0}cl_values/ls.txt'.format(savefolder))
l        = np.loadtxt('{0}cl_values/l.txt'.format(savefolder))


#########################################################################################################
#                                            BARYONIC PLOTS                                             #
#########################################################################################################
# Plot my Cl and camb's Cl for nonlinear growth

plt.figure(6, figsize=(10,6))
plt.clf()
plt.loglog(l , cl_camb, color='k', label=r'$C_\ell^{\kappa\kappa}$ CAMB Function')
plt.loglog(ls, cl_weyl, color='grey', label=r'$C_\ell^{\kappa\kappa}$ CAMB manual original ')
plt.loglog(ls, cl_delta, color='lightgray', label=r'$C_\ell^{\kappa\kappa}$ manual P_weyl(P_delta)')

for i, datakey in enumerate(data_key):
    plt.loglog(lb, cl_baryon_list[i], color=colors[i], label=datakey, ls='dotted')

plt.title(r'$C_\ell^{\kappa\kappa}$ Nonlinear')
plt.ylabel(r'Convergence Power $C_\ell^{\kappa\kappa}$')
plt.xlabel(r'Multipole $\ell$')
plt.grid(True)
plt.xlim(1e2)
plt.legend(loc='center right', ncol=1, bbox_to_anchor=(1.08, 0.70), fancybox=True, prop={'size': 10})
plt.savefig('{0}cl_plots/cl_baryon.pdf'.format(savefolder))

# --------------------------------------------------------------------------------------------------------
# Difference 

# Plot ratio of my Cl and camb's Cl NONLINEAR
plt.figure(7, figsize=(10,6))
plt.clf()

for i, datakey in enumerate(data_key): 
    plt.semilogx(lb, (cl_baryon_list[i]-cl_baryon_list[base_index])/cl_baryon_list[base_index], color=colors[i], label=datakey)    

plt.title(r'Difference $C_\ell^{\kappa\kappa}$ DMONLY vs. $C_\ell^{\kappa\kappa}$ BARYON')
plt.ylabel(r'($C_\ell^{\kappa\kappa, bary}$ - $C_\ell^{\kappa\kappa, DMONLY}$)/$C_\ell^{\kappa\kappa, DMONLY}$')
plt.xlabel(r'Multipole $\ell$')
plt.grid(True)
plt.legend(ncol=2, loc='upper left', prop={'size': 10})
plt.savefig('{0}cl_plots/cl_diff_lmax={1}.pdf'.format(savefolder, lmax))

# --------------------------------------------------------------------------------------------------------
'''
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

# --------------------------------------------------------------------------------------------------------
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