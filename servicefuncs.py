from timeit import default_timer as timer
from typing import List, Any, Iterable
from ismember import ismember
import numpy as np
########################################################################################

def toc(prevtime):
    curtime = timer()
    print('Elaped time {:4f}'.format(curtime - prevtime) + ' seconds')
    return curtime

########################################################################################

def myismember(a,b):
    return ismember(a,b)[0]

########################################################################################

def flatten_recursive(lst: List[Any]) -> Iterable[Any]:
    """Flatten a list using recursion."""
    for item in lst:
        if isinstance(item, list):
            yield from flatten_recursive(item)
        else:
            yield item

#######################################################################

def flattenlist(ll):
    return list(flatten_recursive(ll))

#######################################################################

def formatArray(v):
    vstr = ''
    for jj in range(v.shape[0]):
        vstr = vstr + '{:.4f} '.format(v[jj])
    return vstr

#######################################################################

def findinvec(v,cond): # a Matlab style find function
    return np.arange(v.shape[0])[cond]

#######################################################################

addintercept = lambda XX: np.c_[np.ones((XX.shape[0], 1)), XX]

#######################################################################


#######################################################################

def calclogloss(probMat, y):
    y0prob = probMat[y == 0, 0]
    y1prob = probMat[y == 1, 1]
    if ((y0prob==0).sum()+(y1prob==0).sum()>0):
        logloss = np.inf
    else:
        logloss = -(np.log(y0prob).sum()+np.log(y1prob).sum())/y.size
    return logloss

#######################################################################

def calcbinerrr(probMat, y):
    yhat = probMat.argmax(1)
    return  (1 * (yhat != y)).mean()

#######################################################################