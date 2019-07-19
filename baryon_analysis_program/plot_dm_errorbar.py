import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.interpolate import interp1d
import itertools
mpl.rcParams.update({'font.family':'serif'})
mpl.rcParams.update({'mathtext.fontset':'cm'})
mpl.rcParams.update({'font.size': 18})
mpl.rcParams['lines.linewidth'] = 3
color=itertools.cycle(['k','silver'])

Cls1 = np.loadtxt('data/May21_matter2lens_WF_FDM_1.0_cut_ibarrier_iconc_fCls.csv')
Cls = np.loadtxt('data/May21_matter2lens_WF_CDM_cut_ibarrier_iconc_fCls.csv')

#cdm = interp1d(Cls[:,0],Cls[:,1],bounds_error=True)
fdm = interp1d(Cls1[:,0],Cls1[:,1],bounds_error=True)

nbin = 15

fig = plt.figure(figsize=(9,7))
ax1 = fig.add_subplot(111)
ax2 = ax1.twiny()

l, = ax1.plot(Cls1[:,0],Cls1[:,1],'m',label='$10^{-22}$eV FDM')
l, = ax1.plot(Cls[:,0],Cls[:,1],'c--',label='CDM')
#l.set_dashes(dash1)

Lrange = np.linspace(10000,40000,nbin)
label = ['CMB-HD'] #['0.5 $\mu$K, 20,000 deg$^2$'] 
ls = ['-','--']
shiftFactor = 0. # to shift the errors bar for clarity

noises = [0.5] #0.25
fsky = 0.5

#plt.gca().set_axis_off()
#plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0,hspace = 0, wspace = 0)
#plt.margins(0,0)

for index in range(len(noises)):
    ellBinEdges = np.load('data/190404_experiment_0.25arc_'+str(noises[index])+'uk_2000_reionKSZ_2.912603855229585sqdeg_lbin_edges_dl300.npy')
    ell  =  (ellBinEdges[1:] + ellBinEdges[:-1]) / 2
    cov = np.load('data/190404_experiment_0.25arc_'+str(noises[index])+'uk_2000_reionKSZ_2.912603855229585sqdeg_covmat_dl300.npy')
    covDiag = cov.diagonal() * (2.912603855229585/41253.) / fsky

    # Calculating variance of bin data (weighted mean): 1/var = sum(1/var_i)
    xs,ys = [],[]
    x,y = 0.,0.
    j = 0
    count = 0
    
    for i in range(len(covDiag)):
        
        if Lrange[j] < ell[i]: # end of bin
            if count != 0:
                xs.append(x/count)
                ys.append(1./np.sqrt(y))
                count = 0
            j += 1
            x,y = 0.,0.
            
        count += 1
        x += ell[i]
        y += 1./covDiag[i]
        
    if count != 0:
        xs.append(x/count)
        ys.append(1./np.sqrt(y))
            
    xs = np.array(xs)
    ys = np.array(ys)
        
    #yerr = np.sqrt(cov.diagonal() * (2.91260385523/41253.) / 0.1)
    (_,caps,eb)=ax1.errorbar(xs+shiftFactor,fdm(xs+shiftFactor),yerr=ys,ls='',ecolor=color.next(),capsize=5,label=label[index])
    for cap in caps:
        cap.set_markeredgewidth(2)
    eb[0].set_linestyle(ls[index])
    shiftFactor+=300
        
        
ax1.set_xlabel('$L$',size=24)
ax1.set_ylabel('$C_L^{\kappa\kappa}$',size=24)
ax1.legend(loc='upper right',ncol=1,fontsize=22)
ax1.set_xlim([10000,38000])
ax1.set_ylim([1.3e-11,2.2e-10])
ax1.set_yscale('log')

'''
def tick_function(ell):
    return ell/5000.
ax2.set_xlim(ax1.get_xlim())
#ax2.set_xticks(new_tick_locations)
ax2.set_xticklabels(tick_function(Lrange))
'''

ax1Xs = ax1.get_xticks()

ax2Xs = []
for X in ax1Xs:
    ax2Xs.append(X/5000.)

ax2.set_xticks(ax1Xs)
ax2.set_xbound(ax1.get_xbound())
ax2.set_xticklabels(ax2Xs)
ax2.set_xlabel('$k$ (Mpc$^{-1}$)',size=20)

#plt.show()
plt.savefig('output/dm_errorbar_cmbhd_addaxis.pdf')




#For me: try
# lbins = np.arange(10000, 40000, 232)