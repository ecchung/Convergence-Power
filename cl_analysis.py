#from baryon_analysis import *
import numpy as np
import matplotlib.pyplot as plt
import types
from plot_exp_offset import bottom_offset, top_offset

savefolder   = 'jul08_baryon_analysis/'

    
    
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# For one simulation

# Dweyl simulation
ls = np.arange(2, lmax+1, 500, dtype=np.float64)
nz  = 100
Xmax_cutoff = 2000
Xmin_cutoff = 1000

for li in ls: # fix l
    fig = plt.figure(1, figsize=(9,8))
    plt.clf()
    
    ax1 = fig.add_subplot(311)
    ax2 = fig.add_subplot(323)
    ax3 = fig.add_subplot(324)
    ax4 = ax3.twiny()
    ax5 = fig.add_subplot(325)
    ax6 = fig.add_subplot(326)
    ax7 = ax6.twiny()
    
    #---------------------------------#---------------------------------#---------------------------------
    # a) Plot i vs. X:  (X : 0 -> Xcmb)
    
    Xs = np.linspace(0, Xcmb, nz)
    zs = results.redshift_at_comoving_radial_distance(Xs)
    Xs = Xs[1:-1]
    zs = zs[1:-1]
    As = get_a(zs)
    ws = W(Xs, As, Xcmb)
    d  = np.ones(Xs.shape)
    
    k = (li+0.5) / Xs
    d[:] = 1
    d[k<1e-4] = 0       # change
    d[k>=maxkh] = 0     # change
    
    integrand = d * P_delta(zs, k, from_func='Weyl') * ws**2 / Xs**2  # plot x = Xs, y = integrand
    
    ones = np.ones(integrand.shape)
    
    #---------------------------------
    # Plot a)
    ax1.plot(Xs, integrand, color='orange')
    ax1.plot(Xmin_cutoff*ones, integrand, color='r', label=r'$\chi_{min}$ cutoff'+ ' ({0})'.format(Xmin_cutoff))
    ax1.plot((Xcmb-Xmax_cutoff)*ones, integrand, color='r', label=r'$\chi_{max}$ cutoff'+' ({0})'.format(Xmax_cutoff))
    ax1.set_ylabel('integrand', labelpad=3)
    ax1.set_xlabel(r'$\chi$', labelpad=3)
    ax1.grid(True)
    ax1.legend()
    ax1.tick_params(direction='in')
    ax1.ticklabel_format(axis='both', style='sci', scilimits=(0,0))
    ax1.xaxis._update_offset_text_position = types.MethodType(bottom_offset, ax1.xaxis)
     
    #---------------------------------#---------------------------------#---------------------------------
    # b) Fix Xmin = 0, vary Xmax <~ Xcmb:
    
    Xmin = 0
    Xmax = np.linspace(Xcmb-Xmax_cutoff, Xcmb, 500)
    Xtot = np.linspace(Xmin, Xmax, nz, axis=1)
    cl_dweyl_Xmax = np.zeros(Xmax.shape) # plot x = Xmax, y = cl_dweyl_Xmax
    
    for i, Xs in enumerate(Xtot):
        zs  = results.redshift_at_comoving_radial_distance(Xs)
        dXs = (Xs[2:]-Xs[:-2])/2
        Xs  = Xs[1:-1]
        zs  = zs[1:-1]
        As  = get_a(zs)
        ws  = W(Xs, As, Xcmb)
        d   = np.ones(Xs.shape)
     
        k = (li+0.5) / Xs
        d[:] = 1
        d[k<1e-4] = 0       # change
        d[k>=maxkh] = 0     # change
        
        cl_dweyl_Xmax[i] = np.dot(dXs, d * P_delta(zs, k, from_func='Weyl') * ws**2 / Xs**2)
    
    #---------------------------------
    # Plot b) - Xmax  
    ax2.plot(Xmax, cl_dweyl_Xmax)
    ax2.set_ylabel(r'$C_\ell^{\kappa \kappa}$', labelpad=-2)
    ax2.set_xlabel(r'$\chi_{max}$', labelpad=3)
    ax2.grid(True)
    ax2.tick_params(direction='in')
    ax2.ticklabel_format(axis='both', style='sci', scilimits=(0,0))
    
    
    # Plot b) - (zmax, kmin)
    zmax = results.redshift_at_comoving_radial_distance(Xmax)
    kmin = (li+0.5) / Xmax
    
    ax3.plot(zmax, cl_dweyl_Xmax)
    ax3.set_ylabel(r'$C_\ell^{\kappa \kappa}$', labelpad=-2)
    ax3.set_xlabel(r'$z_{max}$', labelpad=3)
    ax3.grid(True)
    ax3.tick_params(direction='in')
    ax3.ticklabel_format(axis='both', style='sci', scilimits=(0,0))
    ax3.get_yaxis().get_offset_text().set_position((-0.13,0))
    ax3.xaxis._update_offset_text_position = types.MethodType(bottom_offset, ax3.xaxis)
    
    
    ax4.plot(kmin, cl_dweyl_Xmax, color='none') # dummy plot
    ax4.set_xlabel(r'$k_{min}$', labelpad=5)
    ax4.tick_params(direction='in')
    ax4.ticklabel_format(axis='both', style='sci', scilimits=(0,0))
    ax4.xaxis._update_offset_text_position = types.MethodType(top_offset, ax4.xaxis)
    
    #---------------------------------#---------------------------------#---------------------------------
    
    # c) Fix Xmax = Xcmb, vary Xmin >~ 0:
    
    Xmax = Xcmb
    Xmin = np.linspace(0, Xmin_cutoff, 500)
    Xtot = np.linspace(Xmin, Xmax, nz, axis=1)
    cl_dweyl_Xmin = np.zeros(Xmin.shape) # plot x = Xmin, y = cl_dweyl_Xmin
    
    for i, Xs in enumerate(Xtot):
        zs  = results.redshift_at_comoving_radial_distance(Xs)
        dXs = (Xs[2:]-Xs[:-2])/2
        Xs  = Xs[1:-1]
        zs  = zs[1:-1]
        As  = get_a(zs)
        ws  = W(Xs, As, Xcmb)
        d   = np.ones(Xs.shape)
     
        k = (li+0.5) / Xs
        d[:] = 1
        d[k<1e-4] = 0       # change
        d[k>=maxkh] = 0     # change
        
        cl_dweyl_Xmin[i] = np.dot(dXs, d * P_delta(zs, k, from_func='Weyl') * ws**2 / Xs**2)
    
    #---------------------------------
    # Plot c) - Xmin  
    ax5.plot(Xmin, cl_dweyl_Xmin)
    ax5.set_ylabel(r'$C_\ell^{\kappa \kappa}$', labelpad=-2)
    ax5.set_xlabel(r'$\chi_{min}$', labelpad=3)
    ax5.grid(True)
    ax5.tick_params(direction='in')
    ax5.ticklabel_format(axis='both', style='sci', scilimits=(0,0))
    
    # Plot c) - (zmin, kmax)
    zmin = results.redshift_at_comoving_radial_distance(Xmin)
    kmax = (li+0.5) / Xmin
    
    ax6.plot(zmin, cl_dweyl_Xmin)
    ax6.set_ylabel(r'$C_\ell^{\kappa \kappa}$', labelpad=-2)
    ax6.set_xlabel(r'$z_{min}$', labelpad=3)
    ax6.grid(True)
    ax6.tick_params(direction='in')
    ax6.ticklabel_format(axis='both', style='sci', scilimits=(0,0))
    ax6.get_yaxis().get_offset_text().set_position((-0.13,0))
    ax6.xaxis._update_offset_text_position = types.MethodType(bottom_offset, ax6.xaxis)
    
    ax7.plot(kmax, cl_dweyl_Xmin, color='none') # dummy plot
    ax7.set_xlabel(r'$k_{max}$', labelpad=5)
    ax7.tick_params(direction='in')
    ax7.ticklabel_format(axis='both', style='sci', scilimits=(0,0))
    ax7.xaxis._update_offset_text_position = types.MethodType(top_offset, ax7.xaxis)
    
    #---------------------------------
    
    plt.tight_layout(rect=[0,0,1,0.85]) # rect=[x1,y1, x2,y2], pad=#
    plt.subplots_adjust(top=0.88, bottom=0.1, left=0.09, right=0.91, hspace=0.8, wspace=0.25)
    
    # Title
    fig.get_axes()[0].annotate(r'Integrand of $C_\ell (dweyl)$ at $\ell$ = {0}'.format(li), (0.5, 0.93), 
                                xycoords='figure fraction', ha='center', 
                                fontsize=18
                                ) # suptitle won't show up properly
    plt.savefig('{0}integrand_plots/i_dweyl_l={1}.pdf'.format(savefolder, str(li)))
    
    
    