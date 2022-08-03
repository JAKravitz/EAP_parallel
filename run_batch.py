#!/usr/bin/env python
# coding: utf-8
import numpy as np
import pandas as pd
from EAP import EAP
import pickle
import random
import itertools
import time
import pickle
import sys

import dask
import dask.dataframe as dd
from dask.distributed import Client, LocalCluster

def pandafy (array, Deff):
    out = pd.DataFrame(array, index=Deff)
    return out

if __name__ == "__main__":

    cluster = LocalCluster()
    client = Client(cluster)
    
    phytodata = pd.read_csv('/nobackup/jakravit/git/EAP_parallel/phyto_data.csv')
    # phytodata = pd.read_csv('/Users/jakravit/pyProjects/EAP_parallel/phyto_data.csv')
    phytodata = phytodata.iloc[0:4,:]
    # with open('/nobackup/tjnorman/geortm/EAP_02/phyto_dict.pkl', 'rb') as picklefile:
    #     phyto_dict = pickle.load(picklefile)
    
    phytos = phytodata.Species
    phyto_dict = {}
    
    count = 0
    name = 0
    for p in phytos:
        if count%2 == 0:
            name = name+1
            phyto_dict[name] = []
            phyto_dict[name].append(p)
        else:
            phyto_dict[name].append(p)
        count+=1

    for key in phyto_dict.keys():
        print('On node {} are processed: {}'.format(key, phyto_dict[key]))
    # key = int(sys.argv[1])
    #print('the key is: {}'.format(key))

    parameters = ['Qc',
                  'Sigma_c',
                  'cstar',
                  'Qb',
                  'Sigma_b',
                  'bstar',
                  'Qa',
                  'Sigma_a',
                  'astar',
                  'Qbb',
                  'Sigma_bb',
                  'bbstar',]


    # wavelength range and resolution 
    #(changing this changes your interp value when normalising kshell)
    l = np.arange(.4, .901, .001).astype(np.float32) 

    # outpath = '/nobackup/jakravit/data/EAP_batch_outputs/optics_test_s01.p'
    phytodata.info()

    count = 0
    data = {}
    for phyto in phyto_dict[key]:
        #start timer
        start = time.time()

        #add phyto to dictionary
        data[phyto] = {}

        #load phyto data from dataframe
        k = phytodata[phytodata['Species'] == phyto].copy()   
        k.reset_index(inplace=True, drop=True)
        k = k.squeeze()
        im = k.filter(regex='^[0-9]')
        meta = k.filter(regex='^[A-Za-z]+')
        Deff = np.arange(k.Dmin,
                         k.Dmax,
                         k.Dint)
        ncoreX = [1.04]
        nshellX = np.round(np.linspace(k.nshellmin, 
                                       k.nshellmax, 3),2)
        VsX = [.1, .35, .6]
        VeffX = [.6]
        ciX = [2, 3, 5, 7, 9, 12]
        if k.Size_class == 'pico':
            psdX = [np.arange(.2, 10.2, .2)]
        else:
            psdX = [np.arange(1,102,2)] 

        #create iterlist
        iterlist = []
        for it in itertools.product(ncoreX, nshellX, VsX, VeffX, ciX, psdX):
            run = {}
            run['ncore'] = it[0]
            run['nshell'] = it[1]
            run['Vs'] = it[2]
            run['Veff'] = it[3]
            run['ci'] = it[4]
            run['psd'] = it[5]
            iterlist.append(run)

        #create dictionary entries
        for i in range(len(iterlist)):
            rname = '{:.2f}_{:.2f}_{:.2f}_{:.2f}_{}_{}'.format(iterlist[i]['ncore'],
                                                           iterlist[i]['nshell'],
                                                           iterlist[i]['Vs'],
                                                           iterlist[i]['Veff'],                                                                  
                                                           iterlist[i]['ci'],
                                                           max(iterlist[i]['psd']))   
            data[phyto][rname] = iterlist[i]

        for rname in data[phyto].keys():
            print(rname)

            # RUN EAP
            result = dask.delayed(EAP)(l, 
                         im, 
                         Deff, 
                         data[phyto][rname]['ncore'], 
                         data[phyto][rname]['nshell'], 
                         data[phyto][rname]['Vs'], 
                         data[phyto][rname]['Veff'], 
                         data[phyto][rname]['ci']*1e6, 
                         data[phyto][rname]['psd'])
            data[phyto][rname] = result
        data[phyto] = dask.compute(data[phyto])[0]

        # pandafy params so Deff is index
        for rname in data[phyto].keys():
            result = {}
            for param in parameters:
                result[param] = dask.delayed(pandafy)(data[phyto][rname][param], Deff)
            result = dask.compute(result)[0]

            # add run info to dataframe
            result['Deff'] = Deff
            result['ncore'] = iterlist[i]['ncore']
            result['nshell'] = iterlist[i]['nshell']
            result['Vs'] = iterlist[i]['nshell']
            result['Veff'] = iterlist[i]['Veff']
            result['ci'] = iterlist[i]['ci']
            result['psd'] = iterlist[i]['psd']
            result['class'] = meta.Class
            result['PFT1'] = meta.PFT1
            result['PFT2'] = meta.PFT2
            result['size_class'] = meta.Size_class
            result['lambda'] = l

            data[phyto][rname] = result

        #end timer
        end = time.time()
        tleng = end - start
        data[phyto]['time'] = tleng


    for phyto in data.keys():
        print('Phyto {} ran for {:.2f} seconds.'.format(phyto, data[phyto]['time']))

    with open('phytodata_pbs_01.dat', 'wb') as picklefile:
        pickle.dump(data, picklefile)


    #for key in data[phyto]['1.04_1.08_0.10_0.60_2_101'].keys():
    #    print(key, ': {}'.format(data[phyto]['1.04_1.08_0.10_0.60_2_101'][key]))



