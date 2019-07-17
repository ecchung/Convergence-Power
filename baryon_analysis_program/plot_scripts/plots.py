# Just some plots
# Last edited June 26, 2019

import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
import camb
from camb import model, initialpower
from scipy.constants import c           # c / 1000 = 299 km/s
import scipy.integrate as integrate
from scipy import interpolate
import time

savefolder   = 'plots/'

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
l     = np.arange(1,2001)
ls    = np.arange(2, 2001, dtype=np.float64)
lim   = l + 0.5 # limber approximation l
lmax  = 2000

# Set initial paramters using CAMB
pars    = camb.CAMBparams()
pars.set_cosmology(H0 = H0, ombh2=ombh2, omch2=omch2) 
pars.InitPower.set_params(ns = ns)
results = camb.get_background(pars)
Xcmb    = results.conformal_time(0)- results.tau_maxvis
zcmb    = results.redshift_at_comoving_radial_distance(Xcmb)
z       = np.logspace(-5.,np.log10(zcmb), 1500, endpoint=True) 
maxkh   = 2000.
kmax    = 514
npoints = 1000
lpa     = 4  # lens_potential_accuracy

########################################################################################################
#                                        SET MATTER POWER PARAMETERS                                   #
########################################################################################################
PK_linear_f    = camb.get_matter_power_interpolator(pars, nonlinear=False, hubble_units=False, k_hunit=False, kmax=kmax, var1=model.Transfer_tot, var2=model.Transfer_tot, zmax=zcmb)
PK_nonlinear_f = camb.get_matter_power_interpolator(pars, nonlinear=True, hubble_units=False, k_hunit=False, kmax=kmax, var1=model.Transfer_tot, var2=model.Transfer_tot, zmax=zcmb)

z0 = 0
k0 = np.logspace(-1, 0, 1000, endpoint=True)

PK_linear    = PK_linear_f.P(z0, k0)
PK_nonlinear = PK_nonlinear_f.P(z0, k0)


plt.figure(1, figsize=(10,6))
plt.clf()
plt.loglog(k0, PK_linear, color='k', label='Linear')
plt.loglog(k0, PK_nonlinear, color='k', label='Nonlinear', ls= '--')
plt.title(r'Matter Power Spectrum $P_m(k)$ at $z = 0$')
plt.ylabel(r'$P_m(k)$ $[Mpc^{3}/h^{3}]$')
plt.xlabel(r'$k$ $[h/Mpc]$')
plt.grid(True)
plt.legend()
plt.savefig('{0}/Pk.pdf'.format(savefolder))

