import numpy as np
import matplotlib.pyplot as plt
#from mainscript import * # copy and paste while mainscript is running


plt.clf()

plt.figure(1, figsize=(9,6))

plt.loglog(data['TNG100']['k'], data['TNG100']['P'][3], label='TNG100 z='+str(data['TNG100']['z'][3]))
plt.loglog(data['TNG100DM']['k'], data['TNG100DM']['P'][3], label='TNG100DM z='+str(data['TNG100DM']['z'][3]))

plt.loglog(data['TNG300']['k'], data['TNG300']['P'][6], label='TNG300 z='+str(np.round(data['TNG300']['z'][6],3)))
plt.loglog(data['TNG300DM']['k'], data['TNG300DM']['P'][6], label='TNG300DM z='+str(np.round(data['TNG300DM']['z'][6],3)))

plt.title('Illustris TNG Pk with adjustment to TNG300')
plt.xlabel(r'k ($h Mpc^{-1}$)')
plt.ylabel(r'$P(k,z)$')
plt.legend()
plt.show()

plt.clf()

plt.figure(2, figsize=(9,6))

plt.semilogx(lb, cl_allsim_list[8]/TNG100_DMO, label='TNG100 ratio')
plt.semilogx(lb, cl_allsim_list[10]/TNG300_DMO, label='TNG300 ratio')

plt.title('Illustris TNG Clkk ratio with adjustment to TNG300 and Pk ratio')
plt.xlabel(r'$\ell$')
plt.ylabel(r'$C_\ell^{\kappa\kappa}$ bary/DMO ratio')
plt.legend()
plt.show()

plt.clf()