import numpy as np




def minmax_norm(S, min_val, max_val):
    return np.clip((S - min_val) /(max_val - min_val), 0., 1.)


def inv_minmax_norm(x, min_val, max_val):
    return np.clip(x,0,1) * (max_val - min_val)+min_val  