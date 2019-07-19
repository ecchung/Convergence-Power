# BARYONIC ANALYSIS (abstract version of baryon_analysis_old.py)
# Last edited July 17, 2019
import time
beginning = time.time()

import numpy as np
import matplotlib.pyplot as plt
import camb
from camb import model, initialpower
from scipy.constants import c           # c / 1000 = 299 km/s
import scipy.integrate as integrate
from scipy import interpolate
from numpy.linalg import inv
from header.CAMB_header import *
from header.calculate_cl import CAMB_auto, CAMB_Weyl, CAMB_Delta, P_delta, calc_cl_bary
from header.import_baryon_data import import_data, interpolate_ratio

savefolder = 'outputs_jul17/'
datafolder = 'baryonic_data'

colors = np.array(['r', 'darkorange', 'yellow', 'limegreen', 'forestgreen','deepskyblue', 'blue','darkviolet', 'magenta','brown'])

######################################################################################################### 
#                                     IMPORTING BARYONIC DATA                                           #
######################################################################################################### 
data_key, data = import_data()
base_key = 'DMONLY_L100N512'
base_index = int(np.argwhere(data_key==base_key))
data_same = interpolate_ratio(data_key, data, base_index)

'''
######################################################################################################### 
#                                    CALCULATE & SAVE CL VALUES                                         #
######################################################################################################### 
                                #-------------CAMB CL-------------# 
cl_camb  = CAMB_auto(save=True, savefolder=savefolder+'cl_values/', savename='cl_camb')
cl_weyl  = CAMB_Weyl(save=True, savefolder=savefolder+'cl_values/', savename='cl_weyl')
cl_delta = CAMB_Delta(save=True, savefolder=savefolder+'cl_values/', savename='cl_delta')

                                #-----------BARYONIC CL-----------# 
nz = 100
cl_baryon_list = calc_cl_bary(data_key, data_same, base_index, nz, save=True, savefolder=savefolder+'cl_values/', savename='cl_bary_list', R_int=True)
'''

######################################################################################################### 
#                                         IMPORT CL VALUES                                              #
######################################################################################################### 
                                #-------------CAMB CL-------------# 
cl_camb  = np.loadtxt('{0}cl_values/cl_camb.txt'.format(savefolder))
cl_weyl  = np.loadtxt('{0}cl_values/cl_weyl.txt'.format(savefolder))
cl_delta = np.loadtxt('{0}cl_values/cl_delta.txt'.format(savefolder))   

                                #-----------BARYONIC CL-----------# 
cl_baryon_list  = np.loadtxt('{0}cl_values/cl_bary_list.txt'.format(savefolder))

# -------------------------------------------------------------------------------------------------------

######################################################################################################### 
#                                    IMPORT COVARIANCE MATRIX                                           #
######################################################################################################### 

covmat = np.load('covmat.npy')
covinv = inv(covmat)
sigma  = (covmat.diagonal())**2






ending = time.time()
print('Everything took: ', ending - beginning)