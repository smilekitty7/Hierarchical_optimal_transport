## Quoting from the paper: When computing HOTT, we simply truncate LDA topic proportions at 1/(|T | + 1), 
## the value below LDA’s uniform topic proportion prior, and re-normalize. 
def HOTT(u,v,C,prior = -999): ### u and v take LDA topic proportions; weights take the NBOW perhaps
    if prior == -999:
        prior = 1./(len(u) + 1);
    u_new = u[np.where(u > prior)];
    v_new = v[np.where(v > prior)];
    C_new = C[np.where(u > prior)][:,np.where(v > prior)]
    return ot.emd2(u_new,v_new,C_new)



def HOFTT(u,v,C):
    return ot.emd2(u,v,C)
