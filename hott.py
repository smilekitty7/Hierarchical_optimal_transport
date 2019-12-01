from scipy.stats import wasserstein_distance
## Quoting from the paper: When computing HOTT, we simply truncate LDA topic proportions at 1/(|T | + 1), 
## the value below LDA’s uniform topic proportion prior, and re-normalize. 
def HOTT(u,v,u_weight,v_weight,prior = -999): ### u and v take LDA topic proportions; weights take the NBOW perhaps
    if prior == -999:
        prior = 1./(length(u) + 1);
    u_new = u[np.where(u > prior)];
    v_new = v[np.where(v > prior)];
    u_weight_new =u_weight[np.where(u>prior)];
    v_weight_new = v_weight[np.where(v>prior)];
    return wasserstein_distance(u_new,v_new,u_weight_new, v_weight_new)
