## WMD (Word mover's distance)function using scipy.stats.wasserstein_distance:
## parameters are u and v, which are array-like values observed in the (empirical) distribution.
## Weight parameters might be normalized BOW.

### quoting from the paper: Word Mover's distance. Given an embedding of a vocabulary as V ⊂ Rn, the Euclidean metric puts
### a geometry on the words in V . A corpus D = {d1, d2, . . . d|D|} can be represented using distributions
### over V via a normalized BOW.... The WMD between documents d1 and d2 is then WMD(d1, d2) = W1(d1, d2).
from scipy.stats import wasserstein_distance
def wmd(u,v,u_weight,v_weight): ### u and v take two probability distributions; weight perhaps take NBOW
   return wasserstein_distance(u,v,u_weight, v_weight)
#### WMD-T20: find distances between the top 20 words.
import numpy as np
def wmdT20(u,v,u_weight,v_weight): 
    idx_u = np.argsort(u)
    idx_v = np.argsort(v)
    u_t20 = np.zeros(len(indx_u[:20]))
    uw_t20 = np.zeros(len(indx_u[:20]))
    v_t20 = np.zeros(len(indx_v[:20]))
    vw_t20 = np.zeros(len(indx_v[:20]))
    k = 0
    n = 0
    
    for i in idx_u[:20]:
        u_t20[k] = u[i]
        uw_t20[k] = u_weight[i]
        k+=1
    for ii in idx_v[:20]:
        v_t20[n] = v[ii]
        vw_t20[n] = v_weight[ii]
        n+=1

    return wasserstein_distance(u_t20,v_t20,uw_t20, vw_t20)
#### RWMD: code modified from https://github.com/mkusner/wmd/blob/master/compute_rwmd.m
def RWMD(u,v,u_BOW,v_BOW):
    import math
    temp = np.arange(len(u_BOW))
    n = len(u_BOW)
    RWMD_D = np.zeros(n,n)
    DD = np.sum((np.array(u) - np.array(v))**2)
    for i in temp:
        Ei = np.zeros(1,n)
        for j in temp:
            if len(u_BOW[i])>0 or len(v_BOW[i])>0:
                Ei[j]= math.inf
            else:
                x1 = u_BOW[i]/u_BOW[i].sum();
                x2 = v_BOW[i]/v_BOW[i].sum();

                m1 = np.sqrt(max(min(DD[i]),0)); 
                m2 = np.sqrt(max(min(DD[i]),0)); 
                dist1 = m1*x2;
                dist2 = m2*x1;

            Ei[j] = max(dist1,dist2)
            
        RWMD_D[i] = Ei;

    return RWMD_D
# wcd
