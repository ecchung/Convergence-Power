'''  

To be used in mainscript.py                              
                                #----------CALC DELTA CHI----------#                                

clBinned  = np.zeros((len(data_key), len(lBinEdges)-1))  # average the cl values within the bins 
lMidBin   = np.zeros(len(lBinEdges)-1)  # values of l in the middle of the bin
d         = np.ones(lb.shape)

for j, clBaryon in enumerate(cl_baryon_list_lmax1e5):
    cl = clBaryon
    for i in range(len(lBinEdges)-1):
        
        if j == 0: # just do this once
            lMidBin[i] = (lBinEdges[i] + lBinEdges[i+1]) / 2
        
        d[:] = 1  # reset d to all 1
        d[lb<lBinEdges[i]] = 0
        d[lb>=lBinEdges[i+1]] = 0
        cl = clBaryon  # reset cl to original clBaryon
        cl = cl*d
        
        clBinMean = sum(cl) / sum(d)
        clBinned[j][i] = clBinMean # save to ith bin of jth simulation

# Calculate delta chi squared for each simulation
DeltaChi2 = np.zeros(data_key.shape)
clDMO = clBinned[base_index]
for i, clBary in enumerate(clBinned):
    DeltaChi2[i] = sum(((clBary-clDMO)/sigma)**2)
                        
# Calculate delta chi squared using covmat inverse for each simulation
DeltaChi2Inv = np.zeros(data_key.shape)
clDMO = clBinned[base_index]
for i in range(len(data_key)):
    diff = clBinned[i] - clDMO
    DeltaChi2Inv[i] = np.matmul(np.matmul(diff.T, covinv), diff)

    
# Save values
with open('{0}{1}.txt'.format(savefolder+'cl_values/','clBinned'), 'w+') as fclbin:
    np.savetxt(fclbin, clBinned)

with open('{0}{1}.txt'.format(savefolder+'cl_values/','lMidBin'), 'w+') as flmb:
    np.savetxt(flmb, lMidBin)

with open('{0}{1}.txt'.format(savefolder+'cl_values/','DeltaChi2'), 'w+') as fdc:
    np.savetxt(fdc, DeltaChi2)
                                
with open('{0}{1}.txt'.format(savefolder+'cl_values/','DeltaChi2Inv'), 'w+') as fdci:
    np.savetxt(fdci, DeltaChi2Inv)
                                
# Import values
clMeanBin = np.loadtxt('{0}cl_values/clBinned.txt'.format(savefolder))      # (15,) --> bin
lMidBin   = np.loadtxt('{0}cl_values/lMidBin.txt'.format(savefolder))       # (15,) --> bin
DeltaChi2 = np.loadtxt('{0}cl_values/DeltaChi2.txt'.format(savefolder))     # (10,) --> data_key

                                #-----------SIGMA ANALYSIS---------# 

clMidBin = np.zeros((len(data_key), len(lMidBin)))                          # (10, 129)   --> data_key, li
# Extract the Cl values at ell MidBin
for j, l in enumerate(lMidBin):
    for i in range(len(data_key)):
        clMidBin[i,j] = clBary[i,int(lMidBin[j]-10)]

sigmaFrac = sigma/clMidBin[base_index] # fractional sigma                   # (129, ) --> li
'''

''' My failed SN
SN_MidBin  = np.zeros((len(data_key), len(sigma)))
SN_MeanBin = np.zeros((len(data_key), len(sigma)))

SN_MidBin  = np.zeros(data_key.shape)       # (10,) --> data_key
SN_MeanBin = np.zeros(data_key.shape)       # (10,) --> data_key

for i in range(len(data_key)):
    SN_MidBin[i] = np.sqrt(sum((clMidBin[i]/sigma) **2))
    SN_MeanBin[i] = np.sqrt(sum((clMeanBin[i]/sigma) **2))
    
with open('{0}{1}.txt'.format(savefolder+'cl_values/','SN_MidBin'), 'w+') as fmid:
    np.savetxt(fmid, SN_MidBin)
                        
with open('{0}{1}.txt'.format(savefolder+'cl_values/','SN_MeanBin'), 'w+') as fmean:
    np.savetxt(fmean, SN_MeanBin)
'''
