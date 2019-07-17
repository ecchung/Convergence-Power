'''
Used to plot CAMB Cl's.
'''
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate

savefolder   = 'jul08_baryon_analysis/'

######################################################################################################### 
#                                             IMPORT CL VALUES                                          #
######################################################################################################### 
cl_camb  = np.loadtxt('{0}cl_values/cl_camb.txt'.format(savefolder))
cl_weyl  = np.loadtxt('{0}cl_values/cl_weyl.txt'.format(savefolder))
cl_delta = np.loadtxt('{0}cl_values/cl_delta.txt'.format(savefolder))
ls       = np.loadtxt('{0}cl_values/ls.txt'.format(savefolder))
l        = np.loadtxt('{0}cl_values/l.txt'.format(savefolder))


#########################################################################################################
#                                                   PLOTS                                               #
#########################################################################################################

# Plot my Cl and camb's Cl for nonlinear growth
plt.figure(6, figsize=(10,6))
plt.clf()
plt.loglog(l[1:] , cl_camb[1:], color='g', label=r'$C_\ell^{\kappa\kappa}$ CAMB Function')
plt.loglog(ls, cl_weyl, color='b', label=r'$C_\ell^{\kappa\kappa}$ CAMB manual original ')
plt.loglog(ls, cl_delta, color='magenta', label=r'$C_\ell^{\kappa\kappa}$ manual P_weyl(P_delta)')

plt.title(r'$C_\ell^{\kappa\kappa}$ Nonlinear')
plt.ylabel(r'Convergence Power $C_\ell^{\kappa\kappa}$')
plt.xlabel(r'Multipole $\ell$')
plt.grid(True)
plt.legend()
plt.savefig('{0}cl_plots/cl_compare.pdf'.format(savefolder))

# -------------------------------------------------------------------------------------------------------
# Plot ratio of my Cl and camb's Cl NONLINEAR

cl_camb_interp = interpolate.interp1d(l, cl_camb)
cl_camb_int = cl_camb_interp(ls)

plt.figure(7, figsize=(10,6))
plt.clf()

plt.plot(ls[10:], (cl_weyl[10:]-cl_camb_int[10:])/cl_camb_int[10:]*100, color='b', label=r'$\frac{C_\ell^{\kappa\kappa} (CAMB original) - C_\ell^{\kappa\kappa} (camb interp)}{C_\ell^{\kappa\kappa} (camb interp)}$')
plt.plot(ls[10:], (cl_delta[10:]-cl_camb_int[10:])/cl_camb_int[10:]*100, color='magenta', label=r'$\frac{C_\ell^{\kappa\kappa} (dweyl) - C_\ell^{\kappa\kappa} (camb interp)}{C_\ell^{\kappa\kappa} (camb interp)}$')

plt.title(r'Percentage Difference $C_\ell$ mine vs. $C_\ell$ camb')
plt.ylabel(r'Convergence Power $C_\ell^{\kappa\kappa}$ Difference (%)')
plt.xlabel(r'Multipole $\ell$')
plt.grid(True)
plt.ylim(-0.4)
plt.legend(loc='center right', bbox_to_anchor=(0.5, 0.4), fancybox=True)
plt.savefig('{0}cl_plots/cl_perdiff.pdf'.format(savefolder))

