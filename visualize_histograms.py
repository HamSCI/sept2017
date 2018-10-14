#!/usr/bin/python3
import os
import glob
import datetime
import dateutil
from collections import OrderedDict

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import PolyCollection
import cartopy.crs as ccrs

import numpy as np
import pandas as pd
import xarray as xr

import tqdm

import gen_lib as gl

def plot_nc(nc,png_path,param,xkey,xlim=(0,24),ylim=None,log_z=True,**kwargs):
    ds_map  = xr.open_dataset(nc,group='/map')
    ds_data = xr.open_dataset(nc,group='/{!s}'.format(xkey))

    da_map  = ds_map['spot_density']
    da_data = ds_data[param]

    ut_sTime    = dateutil.parser.parse(da_data.attrs['sTime'])
    freqs       = np.sort(da_data['freq_MHz'])[::-1]

    nx      = 100
    ny      = len(freqs)

    fig     = plt.figure(figsize=(30,4*ny))

    plt_nr  = 0
    for inx,freq in enumerate(freqs):
        plt_nr += 1
        ax = plt.subplot2grid((ny,nx),(inx,0),projection=ccrs.PlateCarree(),colspan=30)

        ax.coastlines(zorder=10,color='w')
        ax.plot(np.arange(10))
#        map_data  = da_map.sel(ut_sTime=ut_sTime,freq_MHz=freq).copy()
        map_data  = da_map.sel(freq_MHz=freq).copy()
        map_data  = np.log10(map_data)
        tf        = np.isneginf(map_data)
        map_data.values[tf] = 0
        map_data.plot.contourf(x=da_map.attrs['xkey'],y=da_map.attrs['ykey'],ax=ax,levels=30,cmap=mpl.cm.inferno)
        
        plt_nr += 1
        ax = plt.subplot2grid((ny,nx),(inx,35),colspan=65)
#        data    = da_data.sel(ut_sTime=ut_sTime,freq_MHz=freq).copy()
        data    = da_data.sel(freq_MHz=freq).copy()
        if log_z:
            data        = np.log10(data)
            tf          = np.isneginf(data)
            data.values[tf] = 0

        data.plot.contourf(x=da_data.attrs['xkey'],y=da_data.attrs['ykey'],ax=ax,levels=30)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

    fig.tight_layout()
    fig.savefig(png_path,bbox_inches='tight')
    plt.close(fig)

def main(run_dct):
    src_dir     = run_dct['src_dir']
    baseout_dir = run_dct['baseout_dir']
    xkey        = run_dct['xkey']
    param       = run_dct['param']

    output_dir  = os.path.join(baseout_dir,'{!s}_{!s}_{!s}'.format(os.path.basename(src_dir),param,xkey))
    gl.prep_output({0:output_dir},clear=False)
    ncs = glob.glob(os.path.join(src_dir,'*.nc'))
    ncs.sort()

    for nc in ncs:
        png_path = os.path.join(output_dir,os.path.basename(nc)[:-3]+'_{!s}_{!s}.png'.format(param,xkey))
        plot_nc(nc,png_path,**run_dct)

if __name__ == '__main__':
    baseout_dir = 'output/galleries/histograms'

    run_dcts = []

    rd = {}
    rd['src_dir']       = 'data/histograms/0-10000km_dx30min_dy500km'
    rd['baseout_dir']   = baseout_dir
    rd['xkey']          = 'ut_hrs'
    rd['param']         = 'spot_density'
    run_dcts.append(rd)

    rd = {}
    rd['src_dir']       = 'data/histograms/0-10000km_dx30min_dy500km'
    rd['baseout_dir']   = baseout_dir
    rd['xkey']          = 'slt_mid'
    rd['param']         = 'spot_density'
    run_dcts.append(rd)

    for rd in run_dcts:
        main(rd)
