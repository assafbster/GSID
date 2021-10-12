from timeit import default_timer as timer
import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg as LA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from GMFuncs import hbinentropy, GreedyMerge, hbinary, GMLearn, GMClassify
from sklearn.naive_bayes import BernoulliNB
import scipy.special
from GMclass import GMClassifier
from MakeDataBases import MNISTdbase
from servicefuncs import *

#######################################################################

def CalcTheoreticalLoss(nrep,epsrep,epsingle):
    if nrep*np.log10(epsrep)<-300:
        raise Exception('eps too small - will cause underflow')

    # now calculate the combinatorial channel:
    # input - y, output S = sum y_n (the sufficient statistic)
    PS_y0 = np.zeros(nrep+1)
    for ii in np.arange(nrep+1):
        ncomb = scipy.special.comb(nrep,ii, exact = True)
        PS_y0[ii] = (epsrep**ii)*((1-epsrep)**(nrep-ii))*ncomb
    PS_y0 = PS_y0/sum(PS_y0) # froce normalization sum to one to avoid numeric issues
    PS_y1 = PS_y0[::-1]
    PS = (PS_y0 + PS_y1) / 2. # the marginal distribuion P_S
    Icomb = hbinary(PS) - hbinary(PS_y0)

    Isingle = 1- hbinentropy(np.array([epsingle]))
    # X0 is the single output X0=Y+Z0. Z0\sim Ber(epsingle)
    PSX0_y0 = np.r_[(1-epsingle)*PS_y0,epsingle*PS_y0]
    PSX0_y0 = PSX0_y0/sum(PSX0_y0)
    PSX0_y1 = np.r_[epsingle*PS_y1,(1-epsingle)*PS_y1]
    PSX0_y1 = PSX0_y1/sum(PSX0_y1)
    PSX0 = (PSX0_y0 + PSX0_y1)/2.
    Idouble = hbinary(PSX0) - hbinary(PSX0_y0)

    # Gaussian approximation - used for sanity check
    sigma2 = nrep*(epsrep-epsrep**2)
    sigma = np.sqrt(sigma2)
    ndist = 10001
    xvec = np.linspace(-5*sigma,5*sigma+2*nrep*(0.5-epsrep),ndist)
    PBPSKy0 =  np.exp(-1 / (2 * sigma2) * xvec ** 2)
    PBPSKy0 = PBPSKy0/PBPSKy0.sum()
    PBPSKy1 = PBPSKy0[::-1]
    PBPSK = (PBPSKy0+PBPSKy1)/2.
    IBPSK = hbinary(PBPSK) - hbinary(PBPSKy0)
    loglossAnalytic = (1.-Idouble)/np.log2(np.exp(1.))
    return Icomb, Isingle[0], Idouble, IBPSK, loglossAnalytic

#######################################################################

def MakeRepetitionExample(nsamps,ndim1,ndim2,eps1,eps2):
    y = 1*(np.random.rand(nsamps)>0.5)
    channelnoise1 = 1*(np.random.rand(nsamps,ndim1)<eps1)
    channelnoise2 = 1*(np.random.rand(nsamps)<eps2)
    X1 = np.mod((np.tile(y, (ndim1, 1))).transpose()+channelnoise1,2)
    X2 = np.mod(y+channelnoise2,2)
    X2 = np.tile(X2,(ndim2,1)).transpose()
    X = np.c_[X1,X2]
    indperm = np.random.permutation(ndim1+ndim2)
    X = X[:,indperm]
    return X,y

#######################################################################

def MakeMultiBSC(nsamps,ndim1,ndim2,eps1low,eps1high, eps2):
    y = 1*(np.random.rand(nsamps)>0.5)
    epsvec = np.linspace(eps1low,eps1high,ndim1)
    channelnoise1 = np.zeros((nsamps,ndim1),dtype = int)
    for ii in range(ndim1):
        channelnoise1[:,ii] = 1*(np.random.rand(nsamps)<epsvec[ii])
    X1 = np.mod((np.tile(y, (ndim1, 1))).transpose()+channelnoise1,2)
    channelnoise2 = 1*(np.random.rand(nsamps)<eps2)
    X2 = np.mod(y+channelnoise2,2)
    X2 = np.tile(X2,(ndim2,1)).transpose()
    X = np.c_[X1,X2]
    indperm = np.random.permutation(ndim1+ndim2)
    X = X[:,indperm]
    return X,y

#######################################################################

def MakeRepetitionAR(nsamps,ndim,maineps):
    y = 1*(np.random.rand(nsamps)>0.5)
    channelnoise = 1*(np.random.rand(nsamps,ndim)<maineps)
    X = np.mod((np.tile(y, (ndim, 1))).transpose()+channelnoise,2)
    for ii in np.arange(1,ndim):
        X[:,ii] = np.mod(X[:,ii-1]+X[:,ii],2)
    return X,y

#######################################################################

def MakeLocalDatabase(databaseind, nsamps):
#    databasename = ['MNIST', 'BSCRepetition', 'BSCDouble', 'BSCvariable', 'BSC integrated']
    if (databaseind==0):
        # myMNIST = MNISTdbase(indpermute=False, premseed=0, GMpreprocess=True, Q=2)
        myMNIST = MNISTdbase()
        X_train, y_train, X_test, y_test\
            = myMNIST.makebin([3],[9])

        # X_train, y_train, X_test, y_test\
        #     = myMNIST.makebin([4,5],[7,8])
        # X_train, y_train, X_test, y_test\
        #     = myMNIST.makebin([1, 7, 4, 5, 3, 9],[0, 2, 6, 8])

        nsampstrain, ndim = X_train.shape
        nsampstest, ndim = X_train.shape
        nsamps = nsampstrain+nsampstest
    else:
        if (databaseind in [1,2]): # repetition examples
            if (databaseind==1):
                eps1 = 0.44
                eps2 = 0.5
                ndim1 = 64
                ndim2 = 0
            else:
                eps1 = 0.44
                eps2 = 0.3
                ndim1 = 64
                ndim2 = 64
            X, y = MakeRepetitionExample(nsamps, ndim1, ndim2, eps1, eps2)
            Icomb, Isingle, Idouble, IBPSK, loglossAnalytic = CalcTheoreticalLoss(ndim1, eps1, eps2)
            print('Theoretical values for the data:')
            print('ndim1 = '+str(ndim1)+', ndim2 = '+str(ndim2),', eps1 = '+str(eps1)+', eps2 = '+str(eps2))
            print('I repetition = {:.4f}'.format(Icomb), '[bits],  I single = {:.4f}'.format(Isingle),
                  '[bits]\nI repetition and single  = {:.4}'.format(Idouble),
                  '[bits]\nloglossAnalytic = {:.6f}'.format(loglossAnalytic),'[nats]')
        elif (databaseind==3):
            eps1low  = 0.43
            eps1high = 0.45
            eps2 = 0.3
            ndim1 = 64
            ndim2 = 64
            X, y = MakeMultiBSC(nsamps, ndim1, ndim2, eps1low, eps1high, eps2)
            print('ndim1 = '+str(ndim1)+', ndim2 = '+str(ndim2),
                  ', eps1low = '+str(eps1low)+', eps1high = '+str(eps1high)+', eps2 = '+str(eps2))
        elif (databaseind == 4):
            ndim = 64
            eps = 0.44
            X, y = MakeRepetitionAR(nsamps, ndim, eps)
            print('ndim = '+str(ndim)+', eps = '+str(eps)+', integrated samples')
        X_train, X_test, y_train, y_test = \
            train_test_split(X, y, test_size=.4, random_state=42)
        nsamps, ndim = X_train.shape
    return X_train, X_test, y_train, y_test, nsamps, ndim

#######################################################################
##########   MAIN    ##################################################
#######################################################################
# initialization
np.seterr(all='raise')
np.seterr(under='warn')
starttime = timer()
np.random.seed(0)

# general settings
nsamps = int(1e6) # not applicable for MNIST
M = 32 # the final M of the GM process
databaseinds    = [4]
classifiersinds = [0,1,2,3]

# database and classifier settings
# the classifiers in use
classifiernames = ["SGM, M="+str(M),"RF","NB","Logit","PGM"]
databasename = ['MNIST','BSCRepetition','BSCDouble','BSCvariable','BSC integrated']

#######################################################################

for databaseind in databaseinds:
    print("Database is : " + databasename[databaseind])
    X_trainbinraw, X_testbinraw, y_trainbin, y_testbin, nsamps, ndim = MakeLocalDatabase(databaseind, nsamps)

    #######################################################################

    print('General setting:')
    print('***********************************************************')
    print("nsamps = "+str(nsamps)+ ", ndim = "+str(ndim))
    print('SGM degradation theorey = {:.6f}'.format(ndim*64/M**2)) # print the theoretical degradation

    #######################################################################
    classifiers = [GMClassifier(M, Q=2, Qorig=2, permute=False, permseed=0, Ncoordinates=-1,GreedyOrd=False),
                   RandomForestClassifier(max_depth=10, n_estimators=100, max_features=1, random_state=0),
                   BernoulliNB(),
                   LogisticRegression(max_iter=1000)]
    #######################################################################
    for classifierind in classifiersinds:
        clfname = classifiernames[classifierind]
        clf = classifiers[classifierind]
        print("\nClassifier is : "+clfname)
        # preprocess - turn to binary for "SGM", "RF", "NB"
        if databaseind==0:
            if classifierind in [0,1,2]:# for MNIST, apply quantization
                X_trainbin = 1*(X_trainbinraw>0)
                X_testbin  = 1*(X_testbinraw>0)
            else:
                X_trainbin  = X_trainbinraw.astype(float)/255.
                X_testbin   = X_testbinraw.astype(float)/255.
        else: # the binary databases
            X_trainbin = X_trainbinraw
            X_testbin = X_testbinraw

        curtime = timer()
        clf.fit(X_trainbin, y_trainbin)
        postMattrainGM = clf.predict_proba(X_trainbin)
        postMattestGM  = clf.predict_proba(X_testbin)
        loglosstrainGM = calclogloss(postMattrainGM, y_trainbin)
        loglosstestGM  = calclogloss(postMattestGM, y_testbin)
        trainerrGM = calcbinerrr(postMattrainGM, y_trainbin)
        testerrGM  = calcbinerrr(postMattestGM, y_testbin)
        print('Train: Pe = {:.8f}'.format(trainerrGM),' logloss = {:.8f}'.format(loglosstrainGM))
        print('Test : Pe = {:.8f}'.format(testerrGM),' logloss = {:.8f}'.format(loglosstestGM))
        curtime = toc(curtime)

        if classifierind in [0,4]:
            postMattrainGMall = clf.predict_proba_all(X_trainbin)
            postMattestGMall  = clf.predict_proba_all(X_testbin)
            Niters = postMattrainGMall.shape[0]
            trainerrvec = np.zeros(Niters)
            testerrvec  = np.zeros(Niters)
            testloglossvec  = np.zeros(Niters)
            trainloglossvec = np.zeros(Niters)
            iivec = np.arange(Niters)
            for ii in iivec:
                trainloglossvec[ii] = calclogloss(postMattrainGMall[ii,:,:], y_trainbin)
                testloglossvec[ii]  = calclogloss(postMattestGMall[ii,:,:], y_testbin)
                trainerrvec[ii] = calcbinerrr(postMattrainGMall[ii,:,:], y_trainbin)
                testerrvec[ii]  = calcbinerrr(postMattestGMall[ii,:,:], y_testbin)
            if False:
                print('\n#########################################################')
                print('Train LogLoss per iter =\n' + formatArray(trainloglossvec))
                print('Test  LogLoss per iter =\n' + formatArray(testloglossvec))
                print('Train Error   per iter =\n' + formatArray(trainerrvec))
                print('Test  Error   per iter =\n' + formatArray(testerrvec))
                print('\n#########################################################')

            if False:
                plt.plot(iivec, trainerrvec)
                plt.plot(iivec, testerrvec)
                plt.plot(iivec, trainloglossvec)
                plt.plot(iivec, testloglossvec)
                plt.xlabel('npixel')
                plt.legend(('Pe train','Pe test','logloss train','logloss test'))
                plt.grid(True)
                plt.show()

            if False:
                plt.plot(iivec, trainloglossvec,'k-')
                plt.plot(iivec, testloglossvec,'k-.')
                plt.xlabel('step')
                plt.legend(('logloss train','logloss test'))
                plt.title('M = '+str(M))
                plt.grid(True)
                plt.show()
print('\n***********************************************************')
print('Total run time {:4f}'.format(timer() - starttime) + ' seconds')
