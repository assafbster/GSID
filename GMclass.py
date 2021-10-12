from GMFuncs import *
class GMClassifier():
    def __init__(self,outM=2, Q=2, Qorig = 256, permute = False, permseed = 0, Ncoordinates = -1,GreedyOrd=False):
        self.outM = outM
        self.Qorig = Qorig
        self.Q = Q
        self.GMpreprocess = (self.Qorig>self.Q)
        self.permseed = permseed
        self.permute = permute
        self.Ncoordinates = Ncoordinates
        self.GreedyOrd = GreedyOrd
        if self.GreedyOrd:
            self.Ncoordinates = -1
            self.permute = False


    def GMpre(self,X,y):
        ndims = X.shape[1]
        np.random.seed(self.permseed)
        if self.permute:
            self.permvec = np.random.permutation(ndims)
        elif (self.Ncoordinates==-1):
            self.permvec = np.arange(ndims)
        else:
            self.permvec = np.random.randint(ndims,size = self.Ncoordinates)

        if (self.GMpreprocess & (self.Q<self.Qorig)):
            NY_X0 = np.bincount(X[y==0,:].flatten(), minlength=self.Qorig)
            NY_X1 = np.bincount(X[y==1,:].flatten(), minlength=self.Qorig)
            self.GMquantizer, _, _ = GreedyMerge(NY_X0, NY_X1, self.Qorig, self.Q)
        return

    def preprocess(self,X):
        if (self.GMpreprocess & (self.Q < self.Qorig)):
            X = self.GMquantizer[X]
        X = X[:, self.permvec]
        return X

    def fit(self,X,y):
        self.GMpre(X,y)
        X = self.preprocess(X)
        self.GMbins, self.PX0Ymat, self.Iiter,permvec = GMLearn(X, y, self.outM, self.Q,self.GreedyOrd)
        if self.GreedyOrd:
            self.permvec = permvec
        self.ndim = X.shape[1]

    def GMQuant(self,X):
        ns = X.shape[0]
        Xq = np.zeros((ns,self.ndim),'int')
        XgmPrev = np.zeros(ns, 'int')
        for ii in np.arange(self.ndim):
            XgmCur = XgmPrev * self.Q + X[:, ii]
            XgmPrev = self.GMbins[ii, XgmCur]
            Xq[:,ii] = XgmPrev
        return Xq

    def predict_proba_all(self,X):
        # give the zero posterior probability of
        # all the sequence
        X = self.preprocess(X)
        Xq = self.GMQuant(X)
        ns = X.shape[0]
        PY0seq = np.zeros((self.ndim,ns,2))
        for ii in range(self.ndim):
            curPost = self.PX0Ymat[ii,Xq[:,ii]]
            PY0seq[ii,:,:] = np.c_[curPost, 1-curPost]
        return PY0seq

    def predict_proba(self, X):
        # this is a "black box" function - qunantizing and
        # giving the probability assignment of the last step
        X = self.preprocess(X)
        Xq = self.GMQuant(X)
        PY0_X = self.PX0Ymat[-1,Xq[:,-1]]
        PY_X = np.c_[PY0_X, 1-PY0_X]
        return PY_X

#######################################################################################################################
