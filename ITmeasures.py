import numpy as np

########################################################################################################################

def hbinentropynat(p):
    logp = np.zeros(p.size)
    logp[p>0] = np.log(p[p>0])
    logonep = np.zeros(p.size)
    logonep[1-p > 0] = np.log(1 - p[1-p>0])
    h = -p * logp - (1 - p) * logonep
    return h

########################################################################################################################

def hbinentropy(p):
    log2p = np.zeros(p.size)
    log2p[p>0] = np.log2(p[p>0])
    log2onep = np.zeros(p.size)
    log2onep[1-p > 0] = np.log2(1 - p[1-p>0])
    h = -p * log2p - (1 - p) * log2onep
    return h
########################################################################################################################

def hbinary(p):
    plogp = np.zeros(p.size)
    validinds = np.bitwise_and(p > 0,p<1)
    plogp[validinds] = - p[validinds] * np.log2(p[validinds])
    return plogp.sum()

########################################################################################################################

def hentropynat(p):
    logp = np.zeros(p.size)
    logp[p>0] = np.log(p[p>0])
    logonep = np.zeros(p.size)
    logonep[1-p > 0] = np.log(1 - p[1-p>0])
    h = -p * logp - (1 - p) * logonep
    return h

########################################################################################################################

def calcInfoOut(Qdata, yvec, outM):
    NY_X0 = np.bincount(Qdata[yvec == 0], minlength=outM)
    NY_X1 = np.bincount(Qdata[yvec == 1], minlength=outM)
    PY_X0 = NY_X0 / NY_X0.sum()
    PY_X1 = NY_X1 / NY_X1.sum()
    p0 = yvec.sum()/yvec.shape[0]
    PY = p0 * PY_X0 + (1 - p0) * PY_X1
    Iout = hbinary(PY) - p0 * hbinary(PY_X0) - (1 - p0) * hbinary(PY_X1)
    return Iout