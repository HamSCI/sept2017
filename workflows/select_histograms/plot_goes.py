#!/usr/bin/env python3
import os
import datetime
from collections import OrderedDict

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

import tqdm

import library as lib
from library import goes

def plot_goes(sTime,eTime,output_dir='output',**kwargs):


    goes_dcts       = OrderedDict()
    goes_dcts[13]   = {}
    goes_dcts[15]   = {}

    flares_combined = pd.DataFrame()
    for sat_nr,gd in goes_dcts.items():
        gd['data']      = goes.read_goes(sTime,eTime,sat_nr=sat_nr)
        gd['labels']    = ['GOES {!s}'.format(sat_nr)]
    
#        flares          = goes.find_flares(gd['data'],min_class='M1',window_minutes=60)
#        flares['sat']   = sat_nr
#        gd['flares']    = flares
#        flares_combined = flares_combined.append(flares).sort_index()
    
    fig = plt.figure(figsize=(12,8))
    ax  = fig.add_subplot(1,1,1)

    for sat_nr,gd in goes_dcts.items():
        goes.goes_plot(gd['data'],sTime,eTime,ax=ax,labels=gd['labels'])

    fig.tight_layout()

    dt_str0     = sTime.strftime('%Y%m%d.%H%MUT')
    dt_str1     = eTime.strftime('%Y%m%d.%H%MUT')
    date_str    = '{!s}-{!s}'.format(dt_str0,dt_str1)
    fname       = '{!s}_goes.png'.format(date_str)
    fpath       = os.path.join(output_dir,fname)
    fig.savefig(fpath,bbox_inches='tight')

def main(rd):

    plot_goes(**rd)


if __name__ == '__main__':
    bname       = os.path.basename(__file__)[:-3]
    output_dir  = os.path.join('output/galleries',bname)
    lib.gl.prep_output({0:output_dir})

    rd  = {}
    rd['sTime']  = datetime.datetime(2017,1,1)
    rd['eTime']  = datetime.datetime(2018,1,1)
    rd['output_dir']    = output_dir

    main(rd)
    
import ipdb; ipdb.set_trace()
