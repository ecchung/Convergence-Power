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
    if dirpath == '/mnt/raid-cita/echung/Convergence-Power/baryon_analysis_program':
        datafolder = dirpath+'/baryonic_data/'
    else:
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
        # If it's from Horizon simulation
        if 'Hz' in data_i:
            zval = np.loadtxt(datafolder+data_i, usecols=(1,2,3,4,5,6,7,8,9,10), max_rows=1)
            kval = np.loadtxt(datafolder+data_i, usecols=0, skiprows=4)
            Pval = (np.loadtxt(datafolder+data_i, usecols=(1,2,3,4,5,6,7,8,9,10), skiprows=4)).T   # P[z,k]
            Xval = get_chi(results, zval)
            key  = data_i.strip('_powerspec.out') 
        elif 'TNG' in data_i:
            z, k, P      = np.loadtxt(datafolder+data_i)
            zval, zi     = np.unique(z, return_inverse=True)
            kval, ki     = np.unique(k, return_inverse=True)
            Pval         = np.zeros(zval.shape + kval.shape)
            Pval[zi, ki] = P
            Xval         = get_chi(results, zval)
            key          = data_i.strip('.dat')
        else:
            # Load each data listed in the datafile
            z, k, P      = np.loadtxt('{0}{1}'.format(datafolder, data_i), unpack=True, usecols=(0,1,2), skiprows=1)
            zval, zi     = np.unique(z, return_inverse=True)
            kval, ki     = np.unique(k, return_inverse=True)
            Pval         = np.zeros(zval.shape + kval.shape)
            Pval[zi, ki] = P
            Xval         = get_chi(results, zval)
            if '_L100N512' in data_i:
                data_i = 'OWLS-' + data_i.replace('_L100N512', '')
            if 'txt' in data_i:
                key      = data_i.strip('.txt')
            elif 'dat' in data_i:
                key      = data_i.strip('.dat')
        # Save the values to the data dictionary
        data_key.append(key)
        data[key] = {}
        data[key]['z'] = zval
        data[key]['X'] = Xval
        data[key]['k'] = kval
        data[key]['P'] = Pval
        data[key]['P_interpolator'] = interpolate.interp2d(kval, zval, Pval, kind='cubic')
    print('--------------------------------')
    return np.array(data_key), data

def interpolate_ratio(sim_datakey, data, sim_type, sim_baseindex=None, fix=False):
    print(sim_type)
    if (sim_type == 'BAHAMAS') or (sim_type =='OWLS'):
        sim_data = interpolate_ratio_OWLS_BAHAMAS(sim_datakey, data, sim_baseindex)
    elif sim_type == 'Hz':
        sim_data = interpolate_ratio_Hz(sim_datakey, data, fix=fix)
    elif (sim_type == 'TNG100') or (sim_type == 'TNG300'):
        sim_data = interpolate_ratio_TNG(sim_datakey, data)
    else:
        sim_data = None
    
    return sim_data
        

def interpolate_ratio_OWLS_BAHAMAS(sim_datakey, data, sim_baseindex):
    '''
    return modified data dictionary with same z-values and power spectra ratio interpolator: data_same
    'R_interpolator'
    sim_datakey is the simulation-specific datakey
    sim_base_index is the index of the DMONLY simulation IN THE NEW SIMULATION SPECIFC DATAKEY
    '''
    z_same = np.zeros(10)
    for i, key in enumerate(sim_datakey):
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
    k_same = data[sim_datakey[0]]['k']     # k values are the same for all simulation data
                
    # Fill this dictionary with only the values that correspond to the same z's                
    sim_data = {}
    for i, key in enumerate(sim_datakey):
        sim_data[key] = {}
        sim_data[key]['z'] = z_same
        sim_data[key]['X'] = X_same
        sim_data[key]['k'] = k_same
    
        z_old = data[key]['z']      # Original z
        Pk    = data[key]['P']      # Original P(k)
    
        # Deleting elements of Pk that correspond to z values not in z_same
        #z_old = data[sim_datakey[i]]['z']
        delete_index = []
        for j, zj in enumerate(z_old):
            if zj in z_same:
                pass
            else:
                delete_index.append(j)
        Pk = np.delete(Pk, delete_index, axis=0)
        
        sim_data[key]['P'] = Pk
        sim_data[key]['P_interpolator'] = interpolate.interp2d(k_same, z_same, Pk, kind='cubic')
    
    # Add interpolated ratios of each simulation wrt base_data
    base_same_P = sim_data[sim_datakey[sim_baseindex]]['P'] 
    for i, key in enumerate(sim_datakey):
        bary_same_P = sim_data[key]['P'] 
        sim_data[key]['R_interpolator'] = interpolate.interp2d(k_same, z_same, bary_same_P/base_same_P, kind='cubic', fill_value=1.)
    
    return sim_data


def interpolate_ratio_Hz(sim_datakey, data, fix=False):
    '''
    sim_datakey is simulation-specific datakey
    data is dictionary of ALL data
    '''
    sim_data = {}
    for key in sim_datakey:
        if 'AGN' in key:
            AGN_key = key
            sim_data[key] = data[key]
        elif 'DM' in key:
            DM_key = key
            sim_data[key] = data[key]
    # Interpolate DM to match k and z for AGN
    k        = sim_data[AGN_key]['k']
    z        = sim_data[AGN_key]['z']
    P_AGN    = sim_data[AGN_key]['P']
    P_DM     = sim_data[DM_key]['P']
    P_DM_int = sim_data[DM_key]['P_interpolator'](k,z)
    
    if fix == True: # cheap fix
        for i, zi in enumerate(z):
            print(P_AGN[i,1])
            P_AGN[i] = P_AGN[i] * (P_DM[i,0]/P_AGN[i,0])
            print(P_AGN[i,1])
        sim_data[AGN_key]['R_interpolator'] = interpolate.interp2d(k, z, P_AGN/P_DM, kind='cubic', fill_value=1.)
    elif fix == False: #just do with mismatched z
        sim_data[AGN_key]['R_interpolator'] = interpolate.interp2d(k, z, P_AGN/P_DM, kind='cubic', fill_value=1.)
    elif fix == 'Maybe':# interpolated DM to match k,z with AGN
        sim_data[AGN_key]['R_interpolator'] = interpolate.interp2d(k, z, P_AGN/P_DM_int, kind='cubic', fill_value=1.)
        
    k = sim_data[DM_key]['k']
    z = sim_data[DM_key]['z']
    sim_data[DM_key]['R_interpolator']  = interpolate.interp2d(k, z, sim_data[DM_key]['P']/sim_data[DM_key]['P'], kind='cubic', fill_value=1.)
    
    return sim_data
    
def interpolate_ratio_TNG(sim_datakey, data):
    '''
    sim_datakey is simulation-specific datakey
    data is dictionary of ALL data
    sim_base_index is the index of the DMONLY simulation IN THE NEW SIMULATION SPECIFC DATAKEY
    '''
    sim_data = {}
    for key in sim_datakey:
        if 'DM' in key:
            DM_key = key
            sim_data[key] = data[key]
        else:
            BARY_key = key
            sim_data[key] = data[key]

    # Interpolate DM to match k and z for AGN
    k        = sim_data[BARY_key]['k']
    z        = sim_data[BARY_key]['z']
    P_BARY   = sim_data[BARY_key]['P']
    P_DM     = sim_data[DM_key]['P']
    
    sim_data[BARY_key]['R_interpolator'] = interpolate.interp2d(k, z, P_BARY/P_DM, kind='cubic', fill_value=1.)
            
    k = sim_data[DM_key]['k']
    z = sim_data[DM_key]['z']
    sim_data[DM_key]['R_interpolator']  = interpolate.interp2d(k, z, P_DM/P_DM, kind='cubic', fill_value=1.)
    
    return sim_data
