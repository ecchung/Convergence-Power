
import numpy as np
import matplotlib.pyplot as plt
import types
from plot_exp_offset import bottom_offset, top_offset


fig = plt.figure(figsize=(8,5))
plt.clf()


def f(t):
    return np.exp(-t) * np.cos(2*np.pi*t)

t = np.arange(1, 4, 0.1)
z = t*100000
k = (t+1)/10000

#---------------------------------

ax1 = fig.add_subplot(311)
ax2 = fig.add_subplot(323)
ax3 = fig.add_subplot(324)
ax4 = ax3.twiny()
ax5 = fig.add_subplot(325)
ax6 = fig.add_subplot(326)
ax7 = ax6.twiny()

#---------------------------------
ax1.plot(t*1000000, f(t))
ax1.set_ylabel('integrand', labelpad=0)
ax1.set_xlabel(r'$\chi$', labelpad=3)
ax1.grid(True)
ax1.tick_params(direction='in')
ax1.ticklabel_format(axis='both', style='sci', scilimits=(0,0))
ax1.xaxis._update_offset_text_position = types.MethodType(bottom_offset, ax1.xaxis)
#---------------------------------
ax2.plot(t, f(t))
ax2.set_ylabel(r'$C_\ell^{\kappa \kappa}$', labelpad=-6)
ax2.set_xlabel(r'$X_{max}$', labelpad=3)
ax2.grid(True)
ax2.tick_params(direction='in')
ax2.ticklabel_format(axis='both', style='sci', scilimits=(0,0))


ax3.plot(z, f(t))
ax3.set_ylabel(r'$C_\ell^{\kappa \kappa}$', labelpad=-6)
ax3.set_xlabel(r'$z_{max}$', labelpad=3)
ax3.grid(True)
ax3.tick_params(direction='in')
ax3.ticklabel_format(axis='both', style='sci', scilimits=(0,0))
ax3.get_yaxis().get_offset_text().set_position((-0.13,0))
ax3.xaxis._update_offset_text_position = types.MethodType(bottom_offset, ax3.xaxis)

ax4.plot(k, f(t), color='none')
ax4.set_xlabel(r'$k_{max}$', labelpad=5)
ax4.tick_params(direction='in')
ax4.ticklabel_format(axis='both', style='sci', scilimits=(0,0))
ax4.xaxis._update_offset_text_position = types.MethodType(top_offset, ax4.xaxis)
#---------------------------------
ax5.plot(t, f(t))
ax5.set_ylabel(r'$C_\ell^{\kappa \kappa}$', labelpad=-6)
ax5.set_xlabel(r'$X_{max}$', labelpad=3)
ax5.grid(True)
ax5.tick_params(direction='in')
ax5.ticklabel_format(axis='both', style='sci', scilimits=(0,0))


ax6.plot(z, f(t))
ax6.set_ylabel(r'$C_\ell^{\kappa \kappa}$', labelpad=-7)
ax6.set_xlabel(r'$z_{max}$', labelpad=3)
ax6.grid(True)
ax6.tick_params(direction='in')
ax6.ticklabel_format(axis='both', style='sci', scilimits=(0,0))
ax6.get_yaxis().get_offset_text().set_position((-0.13,0))
ax6.xaxis._update_offset_text_position = types.MethodType(bottom_offset, ax6.xaxis)

ax7.plot(k, f(t), color='none')
ax7.set_xlabel(r'$k_{max}$', labelpad=5)
ax7.tick_params(direction='in')
ax7.ticklabel_format(axis='both', style='sci', scilimits=(0,0))
ax7.xaxis._update_offset_text_position = types.MethodType(top_offset, ax7.xaxis)
#---------------------------------

plt.tight_layout(rect=[0,0,1,0.85]) # rect=[x1,y1, x2,y2], pad=#
plt.subplots_adjust(top=0.88, bottom=0.09, left=0.09, right=0.91, hspace=1.1, wspace=0.25)
fig.get_axes()[0].annotate('Long Suptitle', (0.5, 0.93), 
                            xycoords='figure fraction', ha='center', 
                            fontsize=18
                            ) # suptitle won't show up properly
plt.show()
