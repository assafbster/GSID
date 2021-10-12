# import torch
# import torch.nn as nn
# import torchvision
# import torchvision.transforms as transforms
from timeit import default_timer as timer
# import matplotlib.pyplot as plt
# import addcopyfighandler
import numpy as np
import scipy.io as scipyio
from ITmeasures import *

########################################################################################################################

def quantX(datanp,labels, Q, quantMethod='quantile'):
    # quantMethod = 'uniform'#'uniform','quantile','softquantile'
    nsamps = datanp.size
    dataflat = datanp.flatten()
    if quantMethod == 'uniform':
        datamean = dataflat.mean()
        datastd  = dataflat.std()
        qbins = np.concatenate([[-np.inf],np.linspace(datamean-3*datastd,datamean+3*datastd,Q-1),[+np.inf]])
    elif quantMethod == 'quantile':
        qbins = np.quantile(dataflat,np.linspace(0,1,Q+1))
        qbins[0] = -np.inf
        qbins[-1] = np.inf
    elif quantMethod == 'softquantile':
        sordinds = np.argsort(dataflat)
        dataflat = dataflat[sordinds]
        labels = labels[sordinds]
        Lmin = np.maximum(1, int(0.5 + nsamps / Q))
        Lmax = 2 * Lmin
        qbins = -np.inf*np.ones(Q+1)
        datainds = np.zeros(int(nsamps/Lmin)+1,dtype=int)
        quantDone = False
        ii = 0
        while not(quantDone):
            if (datainds[ii]+Lmin+1>=nsamps):
                datainds[ii] = nsamps-1
                qbins[ii + 1] = np.inf
                quantDone = True
            else:
                curWindow = labels[datainds[ii]:np.minimum(datainds[ii]+Lmax,nsamps)]
                onefrac = curWindow.cumsum()/(1+np.arange(curWindow.size))
                fracbias = np.abs(onefrac[Lmin:]-0.5)
                datainds[ii+1] = datainds[ii]+Lmin+fracbias.size-fracbias[::-1].argmax()
                # the first threshold is always -inf
                if (datainds[ii + 1] == nsamps):
                    qbins[ii + 1] = np.inf
                    quantDone = True
                else:
                    qbins[ii + 1] = (dataflat[datainds[ii+1]-1] + dataflat[datainds[ii + 1]]) / 2
                    ii += 1

    Nqbins = int(1+np.arange(qbins.size)[qbins == np.inf])
    if ((1 * np.iscomplex(datanp)).sum() > 0):
        print('kuku')
    Xquant = np.digitize(datanp,qbins[:Nqbins])-1
    return qbins, Xquant, Nqbins-1

########################################################################################################################

def GreedyMerge(NY_X0,NY_X1,inM,outM):
    inQ = inM / outM
    minVal = np.array([1.e-10])
    NY_X0 = NY_X0 + minVal
    NY_X1 = NY_X1 + minVal
    # consider adding a small value to the zeros
    pY = (NY_X0 + NY_X1)/ (NY_X0 + NY_X1).sum()
    p0 = NY_X0.sum()/(NY_X0.sum() + NY_X1.sum())
    IndsBins = np.arange(inM)
    pX0_Y = (NY_X0+p0)/(NY_X0+NY_X1+1) # the Laplace fix
    # pX0_Y = NY_X0 / (NY_X0 + NY_X1 )
    PY_X0 = pX0_Y * pY / p0
    PY_X1 = (1 - pX0_Y)* pY / (1 - p0)

    postOrd = np.argsort(pX0_Y)
    pX0_Y = pX0_Y[postOrd]
    pY = pY[postOrd]
    IndsVec = np.arange(inM)
    IndsVec = IndsVec[postOrd]
    pYi = pY[:-1]
    pYj = pY[1:]
    p_X0_Yi = pX0_Y[:-1]
    p_X0_Yj = pX0_Y[1:]
    alphaval = pYi / (pYi + pYj)
    p_X0_Yij = alphaval * p_X0_Yi + (1 - alphaval) * p_X0_Yj
    Idiff = (pYi + pYj) * hbinentropynat(p_X0_Yij)\
            -pYi * hbinentropynat(p_X0_Yi) - pYj * hbinentropynat(p_X0_Yj)
    Idiff = np.maximum(Idiff,0) # Idiff is negative only due to numeric errors
# need to limit Idiff by zero - otherwise it's meaningless
    for curM in np.arange(inM-1,outM-1,-1):
        i_ind = Idiff.argmin()
        # merge
        pY[i_ind] = pY[i_ind] + pY[i_ind + 1]
        pY = np.delete(pY,i_ind+1)
        MergedVal1 = IndsBins[IndsVec[i_ind]]
        MergedVal2 = IndsBins[IndsVec[i_ind + 1]]
        IndsBins[IndsBins == MergedVal2] = MergedVal1
        IndsVec = np.delete(IndsVec,i_ind+1)
        mergedPosterior = np.array([p_X0_Yij[i_ind]])
        pX0_Y = np.concatenate([pX0_Y[:i_ind],mergedPosterior,pX0_Y[i_ind + 2:]])
        lowinds = np.arange(max(0, i_ind - 1),min(curM - 1, i_ind+1)) #
        pYi = pY[lowinds]
        pYj = pY[lowinds+1]
        p_X0_Yi = pX0_Y[lowinds]
        p_X0_Yj = pX0_Y[lowinds+1]
        alphaval = pYi / (pYi + pYj)
        p_X0_YijMerged = alphaval * p_X0_Yi + (1 - alphaval) * p_X0_Yj
        IdiffMerged = (pYi + pYj) * hbinentropynat(p_X0_YijMerged)\
                      -pYi * hbinentropynat(p_X0_Yi) - pYj * hbinentropynat(p_X0_Yj)
        Idiff = np.concatenate([Idiff[:max(0,i_ind-1)],IdiffMerged,Idiff[i_ind+2:]])
        p_X0_Yij = np.concatenate([p_X0_Yij[:max(0,i_ind - 1)],p_X0_YijMerged,p_X0_Yij[i_ind + 2:]])

    IndsBinsOut = np.zeros(inM,dtype='int')
    PY_X0out = np.zeros(outM)
    PY_X1out = np.zeros(outM)
    uniqueInds = np.unique(IndsBins)
    for ii in np.arange(uniqueInds.size):
        IndsBinsOut[IndsBins == uniqueInds[ii]] = ii
        PY_X0out[ii] = sum(PY_X0[IndsBins==uniqueInds[ii]])
        PY_X1out[ii] = sum(PY_X1[IndsBins==uniqueInds[ii]])
    PY =  p0*PY_X0out + (1-p0)* PY_X1out
    PX0_Yout = 0.5*np.ones(PY.size)
    PX0_Yout[PY>0] = p0*PY_X0out[PY>0] / PY[PY>0]
    Iout = hbinary(PY) - p0*hbinary(PY_X0out) - (1-p0)*hbinary(PY_X1out)
    return IndsBinsOut, PX0_Yout, Iout

########################################################################################################################

def GMLearn(X, y, outM, Q=2,GreedyOrd=False):
    inM = Q*outM
    ns,ndim = X.shape
    # first iteration
    # GMbins = np.zeros([niters, outM])
    GMbins = np.zeros([ndim, inM],'int')
    PX0Ymat = np.zeros([ndim, outM])
    Iiter = np.zeros(ndim)
    XgmPrev = np.zeros(ns, 'int')
    if not(GreedyOrd):
        for ii in np.arange(ndim):
            XgmCur = XgmPrev*Q+X[:,ii]
            NY_X0 = np.bincount(XgmCur[y == 0], minlength=inM)
            NY_X1 = np.bincount(XgmCur[y == 1], minlength=inM)
            GMbins[ii,:],  PX0Ymat[ii,:], Iiter[ii] = GreedyMerge(NY_X0, NY_X1, inM, outM)
            XgmPrev = GMbins[ii, XgmCur]
        permvec = np.arange(ndim)
    else: # greedy ordering
        indsleft = np.arange(ndim)
        permvec = np.arange(0)
        for ii in np.arange(ndim):
            # find the optimal next index - greedily
            Iinds = np.zeros(indsleft.size)
            for jj in np.arange(indsleft.size):
                XgmCur = XgmPrev * Q + X[:, indsleft[jj]]
                NY_X0 = np.bincount(XgmCur[y == 0], minlength=inM)
                NY_X1 = np.bincount(XgmCur[y == 1], minlength=inM)
                _, _, Iinds[jj] = GreedyMerge(NY_X0, NY_X1, inM, outM)
            optind = indsleft[np.argmax(Iinds)]
            print('ii = '+str(ii)+', optind = '+str(optind)+', I = '+str(np.max(Iinds)))
            permvec = np.r_[permvec, optind]
            indsleft = np.setdiff1d(indsleft,optind)
            # now, repeat the calculation for the optimal ind
            XgmCur = XgmPrev*Q+X[:,optind]
            NY_X0 = np.bincount(XgmCur[y == 0], minlength=inM)
            NY_X1 = np.bincount(XgmCur[y == 1], minlength=inM)
            GMbins[ii,:],  PX0Ymat[ii,:], Iiter[ii] = GreedyMerge(NY_X0, NY_X1, inM, outM)
            XgmPrev = GMbins[ii, XgmCur]
    return GMbins, PX0Ymat, Iiter, permvec

#######################################################################################################################

def GMClassify(X,GMbins, PX0Ymat, outM, Q=2):
    inM = Q*outM
    ns,ndim = X.shape
    XgmPrev = np.zeros(ns, 'int')
    for ii in np.arange(ndim):
        XgmCur = XgmPrev*Q+X[:,ii]
        XgmPrev = GMbins[ii, XgmCur]
    P0out = PX0Ymat[-1, XgmPrev]
    yGM = 1 * (P0out < 0.5)
    return P0out, XgmPrev, yGM

#######################################################################################################################