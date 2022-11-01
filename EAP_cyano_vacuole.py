#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 26 17:06:55 2022

@author: jakravit
"""
#%%
# from numpy import f2py
# sourcefile = open('/Users/jakravit/git/EAP/Dmmex_R14B_4.f','rb')
# sourcecode = sourcefile.read()
# f2py.compile(sourcecode, modulename='Dmmex_R14B_4')
import Dmmex_R14B_4
import numpy as np
import pandas as pd
import pickle
import random
import itertools
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter


def main():
    """The main driver routine."""
    parser = argparse.ArgumentParser(usage="%(prog)s [options]", description='runs a single task group')
    parser.add_argument("--log", type=str, help="A logfile, full path.", default=None)
    # parser.add_argument("work", type=str, help="The work set, i.e. file1, file2", nargs=2)
    args = parser.parse_args()

    logging.basicConfig(filename=args.log, level=logging.DEBUG)

    # fls_log_msg = " ".join([args.work[0], args.work[1]])
    t_start = datetime.datetime.now()
    logging.info("workset start %s", t_start.strftime("%Y%j%H.%M.%S") )
    logging.debug("running EAP MC  ...",)

    # inputs
    Deff = [4,5,6,7]
    # ncore = 1.04
    nshellX = [1.10, 1.14, 1.18] # cv
    VsX = [.3, .5, .7]
    VeffX = [.6]
    ciX = [2e6, 3e6, 5e6]
    psdX = [np.arange(1, 10, .1)]
    PFT1 = 'Cyano_blue'
    PFT2 = 'Cyano_blue'
    PFT3 = 'Cyanophytes'
    size = 'small_nano'
    outpath = '/nobackup/jakravit/data/EAP_optics_MC_v1101.p'
    mcpath = 'Mc_k_shell.csv'
    final = {}

    # hilbert transform (2-layer to equivalent homogenous sphere)
    def analytic_signal(x):
        from scipy.fftpack import fft, ifft
        N = len(x)
        X = fft(x, N)
        h = np.zeros(N)
        h[0] = 1
        h[1:N//2] = 2* np.ones(N// 2-1)
        h[N// 2] = 1
        Z = X * h
        z = ifft(Z, N)
        return z

    def pandafy (array, Deff):
        out = pd.DataFrame(array, index=Deff)
        return out


    ## RUN ##

    for run in itertools.product(nshellX, VsX, VeffX, ciX, psdX):
        #ncore = run[0]
        nshell = run[0]
        Vs = run[1]
        Veff = run[2]
        ci = run[3]
        psd = run[4]

        rname = '{:.2f}_{:.2f}_{:.2f}_{}_10'.format(nshell,
                                                           Vs,
                                                           Veff,                                                                  
                                                           ci/1e6,)


        final[rname] = []
        print (rname)

        # chromatoplasm (shell) imag RI
        im = pd.read_csv(mcpath)
        lshell = im.columns.values.astype(int)[:-1] # wavelength for input RI
        kshell = im.values[0] # imaginary chromatoplasm
        l = np.arange(.4, .901, .001).astype(np.float32) 

        # vacuole RIs as exponential functions (for smoothness)
        ncore = 333.149 * lshell**-1.93855 + 0.8232
        kcore = 22781512.499 * lshell**-4.66 + 1.07719e-5

        # interpolate RI to new resolution
        lshell = lshell /1000
        ncore = griddata(lshell, ncore, l, 'nearest',)
        kcore = griddata(lshell, kcore, l, 'nearest')
        # kcore = 0.041 * np.exp(-11.029 * l)
        kshell_base = griddata(lshell, kshell[:-1], l, 'nearest')

        # define wave parameters
        int_val = np.where(l==.675)[0] # index of 675 nm 
        Vc=1 - Vs
        FR=(1- Vs) ** (1/ 3)# relative volume of core to shell
        nmedia = 1.334
        wavelength = l/ nmedia
        wvno = 2 * np.pi / wavelength # this is a Mie param - combo of size and wavelength
        deltadm=1
        theta901 = np.arange(0, 90.1, 0.1) # length 901

        # shell imag RI
        kshell_norm = (6.75e-7/ nmedia) * (0.027 * ci/ Vs) / (4 * np.pi) #scale to this theoretical max unpackaged chl abs at 675 nm
        kshell = kshell_base * (kshell_norm / kshell_base[int_val])

        # RI's as equivalent sphere using Hilb trans
        nshell = nshell + np.imag(analytic_signal(kshell)) 
        ncore = ncore + np.imag(analytic_signal(kcore))
        khom = kcore*Vc + kshell*Vs # imag RI
        nhom = ncore*Vc + nshell*Vs # real RI
        mshell = nshell - kshell*1j
        mcore = ncore - kcore*1j # chromatoplasm is shell

        # angles for VSF
        nang=901
        angles=np.cos(np.deg2rad(theta901)) # length 901
        theta1 = np.deg2rad(theta901) 
        theta2 = np.flipud(np.pi-np.deg2rad(theta901[0:900]))
        theta=np.append(theta1,theta2)

        # back1=np.where(theta==(np.pi/2)) 
        # back2=np.where(theta==(np.pi))
        d1=np.diff(theta)
        dtheta = np.append(d1, d1[0])

        # preparing variables to be filled
        VSF = np.zeros((len(Deff),len(l), 1801))   #dimensions jjj(deff), nii(wavelength), 1801 angles    
        PF_check = np.zeros((len(Deff),len(l)))
        d_alpha = []  
        PF = np.zeros((len(Deff), len(wavelength), 1801))

        # declare all lists and arrays to fill in the jjj loop (refilled each iteration)
        Qc, Sigma_c, c, Qb, Sigma_b, b, Qa, Sigma_a, a, Qbb, Sigma_bb, bb, bbtilde = (np.zeros([len(Deff),len(l)]) for i in range(13))
        a_sol, a_solm, Qastar2_dir, Qas21 = (np.zeros([len(Deff),len(l)]) for i in range(4))

        for nii in np.arange(0,len(l)): # this is the wavelength loop
            # print(nii)

            # declare lists to be filled on each iteration of the psd loop
            II, phaseMB, alpha, bbprob, bbprob1, Qbbro, checkMB, Qbro, Qcro, M1 = ([] for i in range (10))


            for jj in np.arange(0, len(psd)): # this is the psd loop

                [QEXT,QSCA,GQSC,QBS,m1,m2,s21,d21] = Dmmex_R14B_4.dmilay((psd[jj]*FR)/2,psd[jj]/2,wvno[nii],mshell[nii],mcore[nii],angles,901,901)

                # on each iteration of jj, we get a different QEXT and QSCA out. So these must be stored in their own array
                Qcro.insert(jj,QEXT) 
                Qbro.insert(jj,QSCA)    

                m1_seta = [num[0] for num in m1]
                m1_setb = [num[1] for num in m1]
                M1 = np.append(m1_seta,m1_setb[900:0:-1])
                M2 = np.append(m2[0:901,0], m2[900:0:-1,1])
                myval = (M1+M2)/2 
                II.insert(jj, myval) 

                alpha2=2*np.pi*(psd[jj]/2)/wavelength[nii] 
                alpha.insert(jj, alpha2) 

                phaseMB_jj = [II[jj] / (np.pi* Qbro[jj]* (alpha[jj]**2))]
                phaseMB.insert(jj,phaseMB_jj)

                checkMB_jj = [2* np.pi* np.sum(phaseMB_jj * np.sin(theta) * dtheta)]
                checkMB.insert(jj,checkMB_jj)

                section_jj = [item[900:1801] for item in phaseMB_jj]
                bbprob_jj = 2*np.pi* np.sum((section_jj *np.sin(theta[900:1801]) *dtheta[900:1801]))
                bbprob.insert(jj, bbprob_jj) 
                Qbbro_jj = QSCA * bbprob_jj 
                Qbbro.insert(jj,Qbbro_jj) 

            # we are still in the nii loop here! just the jj loop has ended

            d_alpha_nii = alpha[1] - alpha[0]
            d_alpha.insert(nii,d_alpha_nii)

            # jjj loop starts here
            for jjj in np.arange(0,len(Deff)):

                exponent = (-psd/ 2)/ ((Deff[jjj]/ 2) * Veff)
                psd2 = 1.0e20 * np.power((psd/2),((1-3* Veff)/Veff)) * np.exp(exponent)
                psdm1 = psd / 1e6; 
                psdm2 = psd2 * 1e3; 
                civol = np.pi/ 6 * sum(psdm2 * psdm1 **3 * deltadm)
                psdm2 = psdm2 * (1./ (civol * ci))
                psdvol = np.pi/6 * sum(psdm2 * np.power(psdm1, 3) * deltadm)

                Qc[jjj, nii] = sum(Qcro *psdm2 * np.power(psdm1,2) * deltadm)/ sum(psdm2 * np.power(psdm1,2) *deltadm)
                Sigma_c[jjj,nii] = np.pi/4 * Qc[jjj, nii] * sum(np.power(psdm1, 2) * deltadm)
                c[jjj,nii] = np.pi/4* Qc[jjj, nii]* sum(psdm2* np.power(psdm1,2)* deltadm)

                Qb[jjj, nii] = sum(Qbro * psdm2 * np.power(psdm1,2) * deltadm) /sum(psdm2* np.power(psdm1,2)* deltadm) 	            
                Sigma_b[jjj,nii] = np.pi/4 * Qb[jjj,nii]* sum(np.power(psdm1,2)* deltadm)
                b[jjj, nii] = np.pi/4* Qb[jjj, nii]* sum(psdm2* np.power(psdm1,2)* deltadm)

                Qbb[jjj, nii] = sum(Qbbro * psdm2 * np.power(psdm1,2) * deltadm) /sum(psdm2* np.power(psdm1,2)* deltadm)
                Sigma_bb[jjj, nii] = np.pi/4 * Qbb[jjj, nii] * sum(np.power(psdm1, 2) * deltadm)
                bb[jjj, nii] =  np.pi/4* Qbb[jjj, nii]* sum(psdm2 * np.power(psdm1, 2) * deltadm)

                Qa[jjj, nii] = Qc[jjj, nii] - Qb[jjj, nii]
                Sigma_a[jjj, nii] = np.pi/4 * Qa[jjj, nii]* sum(np.power(psdm1,2)* deltadm)
                a[jjj, nii] = c[jjj, nii] - b[jjj, nii]

                betabar, VSF_1 = ([] for i in range(2))
                checkbar = []
                b_check, bb_check = (np.zeros((len(Deff),len(wavelength))) for i in range(2))

                bbtilde[jjj, nii] = bb[jjj, nii] / b[jjj, nii]

                # this little sub loop is INSIDE the jjj loop		
                for ai in range (0, nang * 2 - 1): # this should go 1801 times - doesn't include the last in range  
                       # need a variable to get(II(:,ai)):
                    varII = [item[ai] for item in II]
                    betabar_ai = (1 / np.pi) * (sum(varII * psdm2 * d_alpha[nii]) / sum(Qbro * psdm2 * np.power(alpha, 2) * d_alpha[nii]))
                    betabar.insert(ai, betabar_ai)
                    VSF_1_ai = betabar[ai] * b[jjj, nii]  
                    VSF_1.insert(ai, VSF_1_ai) # this gives VSF_1 of 1801 angles. For the current instance of nii (wavelength) and jjj (Deff)

                # checkbar is back outside of the sub loop
                checkbar = (2* np.pi * sum(betabar * np.sin(theta) * dtheta))
                PF_check[jjj,nii] = checkbar

                b_check[jjj,nii] = 2 * np.pi * sum((VSF_1) * np.sin(theta) * dtheta)

                PF[jjj,nii,:] = betabar
                VSF[jjj,nii,:] = VSF_1 # VSF_1s are put into matrix on each iteration of Deff, then wavelength.
                # We want to get out all wavelengths but backscatter only angles for each Deff:

                slice_obj = slice(900, 1801) 
                VSF_b = VSF[jjj, nii, slice_obj] # want to get the backward angles for this instance of Deff and all the wavelengths 
                bb_check[jjj,nii] = 2 * np.pi * sum((VSF_b) * np.sin(theta[900: 1801]) * dtheta[900: 1801])      

                ## Package effect calculations:

                ##Set up all phase lag and optical thickness parameters
                ##Calculate acm (absorption celular material) for all layers
                acm_core = (4 * np.pi * kcore[nii]) / (wavelength[nii] * 1e-6)
                acm_shell=(4 * np.pi * kshell[nii]) / (wavelength[nii] * 1e-6)
                acm_hom = (4 * np.pi * khom[nii]) / (wavelength[nii] * 1e-6)
                q = (Deff[jjj] / 2 * FR) / (Deff[jjj] / 2)

                Qas21[jjj, nii] = 3 / 2 * (Qa[jjj, nii] / (acm_core * np.power(q, 3) + acm_shell * (1 - np.power(q, 3)) * 1e-6 * Deff[jjj]))

                ##Direct volume equivalent determination of package effect
                a_sol[jjj, nii] = psdvol * Vc * acm_core + psdvol * Vs * acm_shell
                a_solm[jjj, nii] = psdvol * acm_hom
                Qastar2_dir[jjj,nii] = a[jjj, nii] / a_sol[jjj, nii] #Hayley for input into fluorescence algorithm

                # both the jjj loop and the nii loop end here.

                result = {'Qc': Qc,
                           'Sigma_c': Sigma_c,
                           'cstar': c,
                           'Qb': Qb,
                           'Sigma_b': Sigma_b,
                           'bstar': b,
                           'Qa': Qa,
                           'Sigma_a': Sigma_a,
                           'astar': a,
                           'Qbb': Qbb,
                           'Sigma_bb': Sigma_bb,
                           'bbstar': savgol_filter(bb, 11, 3),
                           'VSF': VSF,
                           'psdvol': psdvol,
                           'VSF_theta': theta,
                           'VSF_angles': np.rad2deg(theta)} 

    ###

        # pandafy params so Deff is index
        for param in ['Qc','Sigma_c','cstar','Qb','Sigma_b','bstar','Qa','Sigma_a','astar','Qbb','Sigma_bb','bbstar',]:
            result[param] = pandafy(result[param], Deff)

        # add run info to dataframe
        result['Deff'] = Deff
        result['ncore'] = ncore
        result['nshell'] = nshell
        result['Vs'] = Vs
        result['Veff'] = Veff
        result['ci'] = ci
        result['psd'] = psd
        result['class'] = clas
        result['PFT1'] = PFT1
        result['PFT2'] = PFT2
        result['size_class'] = size
        result['lambda'] = l

        final[rname] = result

    with open(outpath, 'wb') as fp:
        pickle.dump(final,fp)

    time.sleep(2)
    logging.info("performing post file cleanup...")

    # report on exec time
    t_delt = datetime.datetime.now() - t_start
    logging.info("runtime \"%0.1f\" minutes, seconds %d", float(t_delt.seconds)/60.0, t_delt.seconds) 

    return 0

if __name__ == '__main__':
    sys.exit(main())
