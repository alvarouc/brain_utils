import numpy as np
from scipy.stats import t


def corr2_coeff(A, B):

    # Rowwise mean of input arrays & subtract from input arrays themeselves
    A_mA = A - A.mean(1)[:, None]
    B_mB = B - B.mean(1)[:, None]

    # Sum of squares across rows
    ssA = (A_mA**2).sum(1)
    ssB = (B_mB**2).sum(1)

    # Finally get corr coeff
    return np.dot(A_mA, B_mB.T) / np.sqrt(np.dot(ssA[:, None], ssB[None]))


def corr_p2r(p_value, n_obs):
    # Two tailed test, with N-2 degrees of freedom
    ts = -t.ppf(p_value/2, n_obs - 2)
    r = ts/np.sqrt(ts**2 + n_obs - 2)
    return r
