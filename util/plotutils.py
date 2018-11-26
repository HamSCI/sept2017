import numpy as np

def get_bins(lim, bin_size):
    """ Helper function to split a limit into bins of the proper size """
    return np.arange(lim[0], lim[1]+2*bin_size, bin_size)
