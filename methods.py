## WMD (Word mover's distance)function using scipy.stats.wasserstein_distance:
## parameters are u and v, which are array-like values observed in the (empirical) distribution.
## Weight parameters might be normalized BOW.

### quoting from the paper: Word Mover's distance. Given an embedding of a vocabulary as V ⊂ Rn, the Euclidean metric puts
### a geometry on the words in V . A corpus D = {d1, d2, . . . d|D|} can be represented using distributions
### over V via a normalized BOW.... The WMD between documents d1 and d2 is then WMD(d1, d2) = W1(d1, d2).
from scipy.stats import wasserstein_distance
def wmd(u,v): ### u and v take two probability distributions; weight perhaps take NBOW
   return wasserstein_distance(u,v)

def wmdT20(u,v): 
    idx_u = np.argsort(u)
    idx_v = np.argsort(v)
    u_t20 = np.zeros(len(idx_u[:20]))
    v_t20 = np.zeros(len(idx_v[:20]))
    k = 0
    n = 0
    
    for i in idx_u[:20]:
        u_t20[k] = u[i]

        k+=1
    for ii in idx_v[:20]:
        v_t20[n] = v[ii]

        n+=1

    return wasserstein_distance(u_t20,v_t20)

