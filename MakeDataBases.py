import torchvision
import torchvision.transforms as transforms
import numpy as np
from GMFuncs import *

#######################################################################################################################
####### functions for creating synthetic databases
def make_linearly_separable(nsamps):
    X, y = make_classification(n_samples = nsamps,n_features=2, n_redundant=0, n_informative=2,
                               random_state=1, n_clusters_per_class=1)
    rng = np.random.RandomState(2)
    X += 2 * rng.uniform(size=X.shape)
    ds = (X, y.astype(np.int64))
    return ds

#######################################################################################################################
def make_additive_dbase(nsamps):
    noisestd = 0.2
    ndim = 2
    # X0vals = np.array([[-1, +1],[+1, -1]])
    X0vals = np.array([[-3, +3],[-1,+1],[+1,+3],[+3,+1],[-3,-1],[-1,-3],[+1,-1],[+3,-3]])
    nX0vals = X0vals.shape[0]
    # X1vals = np.array([[-1, -1],[+1, +1]])
    X1vals = np.array([[-3, +1],[-1,+3],[+1,+1],[+3,+3],[-3,-3],[-1,-1],[+1,-3],[+3,-1]])
    nX1vals = X1vals.shape[0]

    y = np.random.randint(2,size=nsamps)
    allinds = np.arange(nsamps)
    y0inds = allinds[y == 0]
    y1inds = allinds[y == 1]
    X = np.zeros((nsamps,ndim))
    X[y0inds, :] = X0vals[np.random.randint(nX0vals, size=y0inds.size),]
    X[y1inds, :] = X1vals[np.random.randint(nX1vals, size=y1inds.size),]
    X = X + noisestd*np.random.randn(nsamps,ndim)
    ds = (X, y.astype(np.int64))
    return ds

#######################################################################################################################

def make_highdimBPSK(nsamps, ndim, targetBPSKInfo, p1, bipolar=False):
    torch.manual_seed(2)
    snrdBvec = -np.random.randint(-10, -0, ndim)
    # now find an offset in dB so that will yield targetBPSKInfo
    noisesigma2vec = 10 ** (-snrdBvec / 10)
    snrMRC = 10*np.log10(np.sum(1/noisesigma2vec))
    snrShannondB = 10*np.log10(2**(2*targetBPSKInfo)-1)
    dBstep = np.array(0.1)
    snroffsetvec = snrShannondB-snrMRC + np.arange(0.,4.,dBstep)
    BPSKcapVec = np.zeros(snroffsetvec.size)
    for ii in range(snroffsetvec.size):
        BPSKcapVec[ii] = calcCBiAWGN(p1,snrMRC+snroffsetvec[ii],bipolar)
        # break from the look when capacity is reached
        # give a higher margin. more tnak 4
    dBOffset = snroffsetvec[np.arange(snroffsetvec.size)[BPSKcapVec > targetBPSKInfo].min()]
    dBOffset = dBstep*np.round(dBOffset/dBstep)
    snrdBvec = snrdBvec+dBOffset
    # on snrdBvec is scaled to provide the required BPSK capacity

    p0 = 1 - p1
    noisesigmavec  = 10 ** (-snrdBvec / 20)
    noisesigma2vec = noisesigmavec**2
    optv = 1/noisesigma2vec
    optv = optv / LA.norm(optv)
    sigma2Proj = sum(noisesigma2vec*optv**2)
    PowerProj = (optv.sum()) ** 2
    optMRCSNRdB = 10 * np.log10(PowerProj / sigma2Proj)

    # generate the data
    y = 1 * (np.random.rand(nsamps) > p0)
    I2opt = calcCBiAWGN(p1, optMRCSNRdB,bipolar)
    if not(bipolar):
        dataBPSK = (np.tile((1 - 2 * y),(ndim,1))).transpose()
    else:
        signvec = (-1)**(1 * (np.random.rand(nsamps) > 0.5))
        dataBPSK = (np.tile( y*signvec, (ndim, 1))).transpose()
    datanoise =  np.random.randn(nsamps,ndim) @ np.diag(noisesigmavec)
    X = dataBPSK + datanoise
    ds = (X, y.astype(np.int64))
    return ds, optv, I2opt, snrdBvec

#######################################################################################################################

def make_MNIST_dbase(zeroclass, oneclass):
    train_dataset = torchvision.datasets.MNIST(root='../../data',
                                               train=True,
                                               transform=transforms.ToTensor(),
                                               download=True)
    test_dataset = torchvision.datasets.MNIST(root='../../data',
                                              train=False,
                                              transform=transforms.ToTensor())

    if np.intersect1d(zeroclass,oneclass).size>0:
        print('Error : sets are not disjoint')
        raise
    allclasses = np.union1d(zeroclass,oneclass)

    # train database
    y_train = train_dataset.targets.numpy()
    validinds = np.in1d(y_train,allclasses)
    y_train = 1*np.in1d(y_train[validinds], oneclass)
    X_train = train_dataset.data.numpy()/255.
    X_train = X_train[validinds,4:-4,4:-4]
    X_train =X_train.reshape((X_train.shape[0],20*20))

    # test database
    y_test = test_dataset.targets.numpy()
    validinds = np.in1d(y_test,allclasses)
    y_test = 1*np.in1d(y_test[validinds], oneclass)
    X_test = test_dataset.data.numpy()/255.
    X_test = X_test[validinds,4:-4,4:-4]
    X_test =X_test.reshape((X_test.shape[0],20*20))
    p1train = y_train.mean()
    str1 = 'zero class ='+format(zeroclass)+'. one class = '+format(oneclass)
    str2 = 'Train Prior = [{:.3f}'.format(1-p1train)+',{:.3f}'.format(p1train)+']. Train corpus size = '+str(y_train.size)
    p1test = y_test.mean()
    str3 = 'Test  Prior = [{:.3f}'.format(1-p1test)+',{:.3f}'.format(p1test)+']. Test  corpus size = '+str(y_test.size)

    return X_train, y_train, X_test, y_test, str1, str2, str3

#######################################################################################################################

class MNISTdbase():
    def __init__(self, indpermute = False, premseed = 0, GMpreprocess = False, Q = 256):
        self.GMpreprocess = GMpreprocess
        self.Q = Q
        self.indpermute = indpermute
        self.premseed = premseed
        train_dataset = torchvision.datasets.MNIST(root='../../data',
                                                   train=True,
                                                   transform=transforms.ToTensor(),
                                                   download=True)
        X_train = np.array(train_dataset.data,dtype='int')
        X_train = X_train[:,4:-4,4:-4].reshape((-1,20 * 20),order='C')
        y_train = np.array(train_dataset.targets,dtype='int')

        test_dataset = torchvision.datasets.MNIST(root='../../data',
                                                  train=False,
                                                  transform=transforms.ToTensor())
        X_test = np.array(test_dataset.data,dtype='int')
        X_test = X_test[:,4:-4,4:-4].reshape((-1,20 * 20),order='C')
        y_test = np.array(test_dataset.targets,dtype='int')
        if self.indpermute:
            np.random.seed(self.premseed)
            self.indperm = np.random.permutation(X_train.shape[1])
            X_train = X_train[:,self.indperm]
            X_test  = X_test[:, self.indperm]

        self.X_train = X_train
        self.X_test  = X_test
        self.y_train = y_train
        self.y_test  = y_test
        return

    def makebin(self,zeroclass, oneclass):
        allclasses = np.union1d(zeroclass, oneclass)
        validinds = np.in1d(self.y_train, allclasses)
        y_trainbin = 1 * np.in1d(self.y_train[validinds], oneclass)
        X_trainbin = self.X_train[validinds, :]
        validinds = np.in1d(self.y_test, allclasses)
        y_testbin  = 1 * np.in1d(self.y_test[validinds], oneclass)
        X_testbin = self.X_test[validinds, :]
        if (self.GMpreprocess & (self.Q<256)):
            NY_X0 = np.bincount(X_trainbin[y_trainbin==0,:].flatten(), minlength=256)
            NY_X1 = np.bincount(X_trainbin[y_trainbin==1,:].flatten(), minlength=256)
            GMquantizer, _, _ = GreedyMerge(NY_X0, NY_X1, 256, self.Q)
            X_trainbin = GMquantizer[X_trainbin]
            X_testbin = GMquantizer[X_testbin]
        return X_trainbin, y_trainbin, X_testbin, y_testbin

    def getdBase(self):
        if (self.GMpreprocess & (self.Q<256)):
            NY_X0 = np.bincount(self.X_train[self.y_train==0,:].flatten(), minlength=256)
            NY_X1 = np.bincount(self.X_train[self.y_train==1,:].flatten(), minlength=256)
            GMquantizer, _, _ = GreedyMerge(NY_X0, NY_X1, 256, self.Q)
            X_train = GMquantizer[self.X_train]
            X_test = GMquantizer[self.X_test]
        else:
            X_train = self.X_train
            X_test  = self.X_test
        return X_train, self.y_train, X_test, self.y_test

#######################################################################################################################