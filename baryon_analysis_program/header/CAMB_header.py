'''
Contains necessary parameter settings and function definitions.
Should import * when in use.l
Set parameters as required.
'''

import numpy as np
import camb
from camb import model, initialpower
from scipy.constants import c           # c / 1000 = 299 km/s


# Define parameters used (values from "Planck alone" column of
# Table 6 of https://arxiv.org/pdf/1807.06205.pdf)
H0    = 67.3            # Hubble constant today         (km/s/Mpc)
H0c   = H0/c * 1000     # Hubble constant today         (1/Mpc)
h     = H0/100          # H0 / 100 km/s /Mpc            (unitless)
ombh2 = 0.0224          # baryonic matter density       (unitless)
omch2 = 0.1201          # cold dark matter density      (unitless)
omb   = ombh2 / h**2    # baryonic matter density fraction  (unitless)
omc   = omch2 / h**2    # cold dark matter density fraction (unitless)
omm   = omb + omc       # total matter density          (unitless)
oml   = 1 - (omb + omc) # omega lambda                  (unitless)
omk   = 0.0             # curvature parameter           (unitless)
ns    = 0.966           # scalar spectral index         (unitless)
As    = 2.1e-9          # amplitude of scalar spectrum  (unitless)
piv   = 0.05            # scale at which As is defined  (1/Mpc)

lmax  = 5000
l     = np.arange(1,lmax+1)
ls    = np.arange(2, lmax+1, dtype=np.float64)
lim   = l + 0.5 # limber approximation l


# Set initial paramters using CAMB
pars    = camb.CAMBparams()
pars.set_cosmology(H0 = H0, ombh2=ombh2, omch2=omch2)
pars.InitPower.set_params(ns = ns, As = As, pivot_scalar = piv)
results = camb.get_background(pars)
Xcmb    = results.conformal_time(0)- results.tau_maxvis
zcmb    = results.redshift_at_comoving_radial_distance(Xcmb)
maxkh   = 2000.
npoints = 1000
z       = np.logspace(-5.,np.log10(zcmb), 1500, endpoint=True)


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


