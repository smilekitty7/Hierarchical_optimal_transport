## parameters are u and v, which are array-like values observed in the (empirical) distribution.
## Cost might be normalized BOW.
import ot

def wmd(u,v): 
    u = np.ascontiguousarray(u)
    v = np.ascontiguousarray(v)
   return ot.emd2(u,v,C)

def wmdt20(u, v, C):
    idx_u = np.argsort(u)
    idx_v = np.argsort(v)
    u_t20 = np.zeros(len(idx_u[-20:]))
    v_t20 = np.zeros(len(idx_v[-20:]))
    k = 0
    n = 0
    for i in idx_u[-20:]:
        u_t20[k] = u[i]
        k+=1
    for ii in idx_v[-20:]:
        v_t20[n] = v[ii]
        n+=1
    u_new = np.ascontiguousarray(u_t20)
    v_new = np.ascontiguousarray(v_t20)
    return ot.emd2(u_new, v_new, C)

## Quoting from the paper: When computing HOTT, we simply truncate LDA topic proportions at 1/(|T | + 1), 
## the value below LDA’s uniform topic proportion prior, and re-normalize. 
def HOTT(u,v,C,prior = -999):### u and v take LDA topic proportions (truncate); not sure about the cost.
    t = len(u)
    if prior is -999:
        prior = 1. / (t + 1)
    idx_p = u > prior
    idx_q = v > prior
    u_new = u[idx_p]
    v_new = v[idx_q]
    u_new = np.ascontiguousarray(u_new)
    v_new = np.ascontiguousarray(v_new)
    return ot.emd2(u_new, v_new, C)

def HOFTT(u,v,C): #### u and v take LDA topic proportions (no truncate); not sure about the cost.
    u = np.ascontiguousarray(u)
    v = np.ascontiguousarray(v)
    return ot.emd2(u,v,C)

