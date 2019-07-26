import numpy as np
import matplotlib.pyplot as plt
import types
import sys, os
sys.path.append(os.path.abspath('..')) # now in baryon_analysis_program folder
dirpath = os.getcwd() 
from plot_scripts.plot_exp_offset import bottom_offset, top_offset
from header.CAMB_header import *
from header.CAMB_cl_calc import P_delta
from mainscript import data_key, data, data_same, z_same, base_index, colors

savefolder   = 'jul08_baryon_analysis/'
lmax = 1e5

# Bary simulation    
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
'''
##########################################################################################################
#                               Plotting Integrand and Cl vs. Xmax and Xmin                              #
##########################################################################################################
ls = np.arange(2, lmax+1, 1000, dtype=np.float64)
nz = 100
Xmax_cutoff = 2000
Xmin_cutoff = 1000
k_max = 514.71854     # largest k value from all simulations
k_min = 0.062831853   # smallest k value from all simulations

for j, datakey in enumerate(data_key[7:]):
    print(j, datakey)
    P_ratio_int = data_same[datakey]['P_ratio_interpolator']
    
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
        d[k>=k_max] = 0     # change
        
        integrand =  P_delta(zs, k, from_func='Weyl') * np.diagonal(np.flip(P_ratio_int(k, zs), axis=1)) * ws**2 / Xs**2 
         # plot x = Xs, y = integrand
        
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
        cl_bary_Xmax = np.zeros(Xmax.shape) # plot x = Xmax, y = cl_dweyl_Xmax
        
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
            d[k>=k_max] = 0     # change
            
            cl_bary_Xmax[i] = np.dot(dXs, d * P_delta(zs, k, from_func='Weyl') * np.diagonal(np.flip(P_ratio_int(k, zs), axis=1)) * ws**2 / Xs**2) 
        
        #---------------------------------
        # Plot b) - Xmax  
        ax2.plot(Xmax, cl_bary_Xmax)
        ax2.set_ylabel(r'$C_\ell^{\kappa \kappa}$', labelpad=-2)
        ax2.set_xlabel(r'$\chi_{max}$', labelpad=3)
        ax2.grid(True)
        ax2.tick_params(direction='in')
        ax2.ticklabel_format(axis='both', style='sci', scilimits=(0,0))
        
        
        # Plot b) - (zmax, kmin)
        zmax = results.redshift_at_comoving_radial_distance(Xmax)
        kmin = (li+0.5) / Xmax
        
        ax3.plot(zmax, cl_bary_Xmax)
        ax3.set_ylabel(r'$C_\ell^{\kappa \kappa}$', labelpad=-2)
        ax3.set_xlabel(r'$z_{max}$', labelpad=3)
        ax3.grid(True)
        ax3.tick_params(direction='in')
        ax3.ticklabel_format(axis='both', style='sci', scilimits=(0,0))
        ax3.get_yaxis().get_offset_text().set_position((-0.13,0))
        ax3.xaxis._update_offset_text_position = types.MethodType(bottom_offset, ax3.xaxis)
        
        
        ax4.plot(kmin, cl_bary_Xmax, color='none') # dummy plot
        ax4.set_xlabel(r'$k_{min}$', labelpad=5)
        ax4.tick_params(direction='in')
        ax4.ticklabel_format(axis='both', style='sci', scilimits=(0,0))
        ax4.xaxis._update_offset_text_position = types.MethodType(top_offset, ax4.xaxis)
        
 #---------------------------------#---------------------------------#---------------------------------
        
        # c) Fix Xmax = Xcmb, vary Xmin >~ 0:
        
        Xmax = Xcmb
        Xmin = np.linspace(0, Xmin_cutoff, 500)
        Xtot = np.linspace(Xmin, Xmax, nz, axis=1)
        cl_bary_Xmin = np.zeros(Xmin.shape) # plot x = Xmin, y = cl_bary_Xmin
        
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
            d[k>=k_max] = 0     # change
            
            cl_bary_Xmin[i] = np.dot(dXs, d * P_delta(zs, k, from_func='Weyl') * np.diagonal(np.flip(P_ratio_int(k, zs), axis=1)) * ws**2 / Xs**2) 
        
        #---------------------------------
        # Plot c) - Xmin  
        ax5.plot(Xmin, cl_bary_Xmin)
        ax5.set_ylabel(r'$C_\ell^{\kappa \kappa}$', labelpad=-2)
        ax5.set_xlabel(r'$\chi_{min}$', labelpad=3)
        ax5.grid(True)
        ax5.tick_params(direction='in')
        ax5.ticklabel_format(axis='both', style='sci', scilimits=(0,0))
        
        # Plot c) - (zmin, kmax)
        zmin = results.redshift_at_comoving_radial_distance(Xmin)
        kmax = (li+0.5) / Xmin
        
        ax6.plot(zmin, cl_bary_Xmin)
        ax6.set_ylabel(r'$C_\ell^{\kappa \kappa}$', labelpad=-2)
        ax6.set_xlabel(r'$z_{min}$', labelpad=3)
        ax6.grid(True)
        ax6.tick_params(direction='in')
        ax6.ticklabel_format(axis='both', style='sci', scilimits=(0,0))
        ax6.get_yaxis().get_offset_text().set_position((-0.13,0))
        ax6.xaxis._update_offset_text_position = types.MethodType(bottom_offset, ax6.xaxis)
        
        ax7.plot(kmax, cl_bary_Xmin, color='none') # dummy plot
        ax7.set_xlabel(r'$k_{max}$', labelpad=5)
        ax7.tick_params(direction='in')
        ax7.ticklabel_format(axis='both', style='sci', scilimits=(0,0))
        ax7.xaxis._update_offset_text_position = types.MethodType(top_offset, ax7.xaxis)
        
        #---------------------------------
        
        plt.tight_layout(rect=[0,0,1,0.85]) # rect=[x1,y1, x2,y2], pad=#
        plt.subplots_adjust(top=0.88, bottom=0.1, left=0.09, right=0.91, hspace=0.8, wspace=0.25)
        
        # Title
        fig.get_axes()[0].annotate(r'Integrand of $C_\ell (bary)$ at $\ell$ = {0}'.format(li), (0.5, 0.93), 
                                    xycoords='figure fraction', ha='center', 
                                    fontsize=18
                                    ) # suptitle won't show up properly
        plt.savefig('{0}integrand_plots/{1}/i_bary{1}_l={2}.pdf'.format(savefolder, datakey, str(li)))
'''    
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

##########################################################################################################
#                                  Plotting Cl vs.l for different Zmax                                   #
##########################################################################################################
# Varying the z_max
z_max = np.flip(z_same)[:10]
X_max = get_chi(results, z_max)

nz  = 100
Xb  = np.linspace(0, Xcmb, nz)      # full range
zb  = results.redshift_at_comoving_radial_distance(Xb)
kmax= 514.71854                     # largest k value from all simulations
kmin= 0.062831853                   # smallest k value from all simulations

dXb = (Xb[2:]-Xb[:-2])/2
Xb  = Xb[1:-1]
zb  = zb[1:-1]
Ab  = get_a(zb)

# Get lensing window function (flat universe)
wb = W(Xb, Ab, Xcmb)
d  = np.ones(Xb.shape)

L = np.arange(10, lmax+1, 500, dtype=np.float64)

def R(ratio_int, k,z, zmax):
    '''
    k -> (ndarray)
    z -> (ndarray)
    zmax -> (float)
    '''
    if z<zmax:
        return ratio_int(k, z)
    else:
        return 1

R_vec = np.vectorize(R)
    
# Calculate the integral

# For DMONLY
datakey = data_key[base_index]
P_ratio_int = data_same[datakey]['P_ratio_interpolator']
cl_base_ALLzmax = []

# For each zmax
for j, Xmax in enumerate(X_max):
    cl_bary_zmax = np.zeros(L.shape)
    zmax = z_max[j]
    
    # For each l
    for i, li in enumerate(L):
        k = (li+0.5) / Xb
        d[:] = 1
        d[k>=kmax] = 0
        
        # the interpolator weirdly sorts P values into increasing values of k
        #cl_bary_zmax[i] = np.dot(dXb, d * P_delta(zb, k, from_func='Weyl') * np.diagonal(np.flip(P_ratio_int(k, zb), axis=1)) * wb**2 / Xb**2) 
        
        cl_bary_zmax[i] = np.dot(dXb, d * P_delta(zb, k, from_func='Weyl') * R_vec(P_ratio_int, k,zb, zmax) * wb**2 / Xb**2)
    
    # Save it to All zmax
    cl_base_ALLzmax.append(cl_bary_zmax)
   
  
  
        

# For each bary sim
for datakey in data_key:
    print(datakey)
    P_ratio_int = data_same[datakey]['R_interpolator']
    
    fig1, ax1 = plt.subplots(figsize=(10,6))
    fig2, ax2 = plt.subplots(figsize=(10,6))
    fig3, ax3 = plt.subplots(figsize=(10,6))
    
    ax1.set_title(datakey + '\n' + r'$C_\ell$ norm ratio at $z_{max}$')
    ax2.set_title(datakey + '\n' + r'$C_\ell$ ratio at $z_{max}$')
    ax3.set_title(datakey + '\n' + r'$C_\ell$ ratio difference at $z_{max}$')
    
    # For each zmax
    for j, Xmax in enumerate(X_max):
        cl_bary_zmax = np.zeros(L.shape)
        zmax = z_max[j]
        cl_bary_z6 = np.zeros(L.shape)
        
        # For each l
        for i, li in enumerate(L):
            k = (li+0.5) / Xb
            d[:] = 1
            d[k>=kmax] = 0
            
            # the interpolator weirdly sorts P values into increasing values of k
            #cl_bary_zmax[i] = np.dot(dXb, d * P_delta(zb, k, from_func='Weyl') * np.diagonal(np.flip(P_ratio_int(k, zb), axis=1)) * wb**2 / Xb**2) 
            
            cl_bary_zmax[i] = np.dot(dXb, d * P_delta(zb, k, from_func='Weyl') * R_vec(P_ratio_int, k,zb, zmax) * wb**2 / Xb**2)
        
        if j == 0:
            cl_bary_z6 = cl_bary_zmax
        else:
            ax2.semilogx(L, (cl_bary_zmax - cl_bary_z6)/cl_base_ALLzmax[j], color=colors[j], label=r'$z_{max}$' + '= {0}'.format(zmax))
            
        ax1.semilogx(L, (cl_bary_zmax-cl_base_ALLzmax[j])/cl_base_ALLzmax[j], color=colors[j], label=r'$z_{max}$' + '= {0}'.format(zmax))
        ax3.semilogx(L, (cl_bary_zmax)/cl_base_ALLzmax[j], color=colors[j], label=r'$z_{max}$' + '= {0}'.format(zmax))
    
    ax1.legend()
    ax1.set_ylabel(r'($C_\ell^{\kappa\kappa, bary}$ - $C_\ell^{\kappa\kappa, DMONLY}$)/$C_\ell^{\kappa\kappa, DMONLY}$')
    ax1.set_xlabel(r'$\ell$')
    ax1.grid(True)
    #fig1.show()
    #fig1.savefig('{0}zmax_analysis/{1}_normratio_vs_zmax.pdf'.format(savefolder, datakey))
    
    ax2.legend()
    ax2.set_ylabel(r'$\frac{C_\ell^{\kappa\kappa, bary}(z_{max}) - C_\ell^{\kappa\kappa, bary}(z_{max}=6)}{C_\ell^{\kappa\kappa, DMONLY}(z_{max}=6)}$')
    ax2.set_xlabel(r'$\ell$')
    ax2.grid(True)
    #fig2.show()
    fig2.savefig('{0}zmax_analysis/{1}_ratio_diff_vs_zmax.pdf'.format(savefolder, datakey))

    ax3.legend()
    ax3.set_ylabel(r'$C_\ell^{\kappa\kappa, bary}$/$C_\ell^{\kappa\kappa, DMONLY}$')
    ax3.set_xlabel(r'$\ell$')
    ax3.grid(True)
    #fig3.show()
    #fig3.savefig('{0}zmax_analysis/{1}_ratio2_vs_zmax.pdf'.format(savefolder, datakey))

    
 