# IMPORTING BARYONIC DATA

import numpy as np
from scipy import interpolate
from header.CAMB_header import *
import os
dirpath = os.getcwd()

def import_data():
    '''
    Returns list of data keys and a data dictionary with keys: data_key, data
    'z'
    'X'
    'k'
    'P'
    'P_interpolator'
    '''
    # Load data file names
    datafolder = dirpath+'/baryonic_data/'
    if __name__ == '__main__':
        datafolder = dirpath+'/../baryonic_data/'
    datafile = np.loadtxt(datafolder+'data.txt', dtype=str)
    
    # Load data into a dictionary accessible with keys that are name of the files without .txt and 'z', 'X', 'k', 'P'
    data = {}
    data_key = []
    
    print('--------------------------------')
    print('Importing Baryonic Data...')
    print('i datafilename')
    for i, data_i in enumerate(datafile):
        print(i, data_i)
        
        # Load each data listed in the datafile
        z, k, P = np.loadtxt('{0}{1}'.format(datafolder, data_i), unpack=True, usecols=(0,1,2), skiprows=1)
        
        zval, zi     = np.unique(z, return_inverse=True)
        kval, ki     = np.unique(k, return_inverse=True)
        Pval         = np.zeros(zval.shape + kval.shape)
        Pval[zi, ki] = P
        Xval         = get_chi(results, zval)
        
        # Save the values to the data dictionary
        key = data_i.strip('.txt')
        data_key.append(key)
        data[key] = {}
        data[key]['z'] = zval
        data[key]['X'] = Xval
        data[key]['k'] = kval
        data[key]['P'] = Pval
        data[key]['P_interpolator'] = interpolate.interp2d(kval, zval, Pval, kind='cubic')
    print('--------------------------------')
    
    return np.array(data_key), data
    
    
def interpolate_ratio(data_key, data, base_index):
    '''
    return modified data dictionary with same z-values and power spectra ratio interpolator: data_same
    'R_interpolator'
    '''
    z_same = np.zeros(10)
    for i, key in enumerate(data_key):
        z_vals = data[key]['z']
        if i == 0:      # Set first z's as Z
            z_same = z_vals
        else:
            for j, zj in enumerate(z_same):
                if zj in z_vals:
                    pass
                else:
                    z_same = np.delete(z_same, j)

    X_same = get_chi(results, z_same)
    k_same = data[data_key[0]]['k']     # k values are the same for all simulation data
                
    # Fill this dictionary with only the values that correspond to the same z's                
    data_same = {}
    for i, key in enumerate(data_key):
        data_same[key] = {}
        data_same[key]['z'] = z_same
        data_same[key]['X'] = X_same
        data_same[key]['k'] = k_same
    
        z_old = data[key]['z']      # Original z
        Pk    = data[key]['P']      # Original P(k)
    
        # Deleting elements of Pk that correspond to z values not in z_same
        z_old = data[data_key[i]]['z']
        delete_index = []
        for j, zj in enumerate(z_old):
            if zj in z_same:
                pass
            else:
                delete_index.append(j)
        Pk = np.delete(Pk, delete_index, axis=0)
        data_same[key]['P'] = Pk
    
        data_same[key]['P_interpolator'] = interpolate.interp2d(k_same, z_same, Pk, kind='cubic')
    
    # Add interpolated ratios of each simulation wrt base_data
    base_same_P = data_same[data_key[base_index]]['P'] 
    for i, key in enumerate(data_key):
        bary_same_P = data_same[key]['P'] 
        data_same[key]['R_interpolator'] = interpolate.interp2d(k_same, z_same, bary_same_P/base_same_P, kind='cubic', fill_value=1.)
    
    return data_same
    
    
    
    
