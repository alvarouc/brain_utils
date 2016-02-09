import numpy as np
from scipy.stats import t


def corr_p2r(p_value, n_obs):
    # Two tailed test, with N-2 degrees of freedom
    ts = -t.ppf(p_value/2, n_obs - 2)
    r = ts/np.sqrt(ts**2 + n_obs - 2)
    return r
