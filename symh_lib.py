#!/usr/bin/env python3
import os
import datetime

#import matplotlib as mpl
#mpl.use('Agg')
#from matplotlib import pyplot as plt

import pandas as pd
import numpy as np

def load_symh(fpath='data/kyoto_wdc/WWW_aeasy00025746.dat.txt'):
    df  = pd.read_csv(fpath,sep='\s+',header=14,parse_dates=[['DATE','TIME']])
    df  = df.set_index('DATE_TIME')
    df  = df[['ASY-D','ASY-H','SYM-D','SYM-H']].copy()

    return df

if __name__ == '__main__':
    df      = load_symh()
    symh    = df['SYM-H']

    dt_0    = datetime.datetime(2017,9,12)
    dt_1    = datetime.datetime(2017,9,14)

    tf      = np.logical_and(symh.index >= dt_0, symh.index < dt_1)
    dft     = symh[tf].copy()
    import ipdb; ipdb.set_trace()
