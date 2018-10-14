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
import netCDF4

import tqdm

import gen_lib as gl

pdict   = {}

#dct = {}
#dct['log_z']    = False
#pdict['mean']   = dct
#
#dct = {}
#dct['log_z']    = False
#pdict['median'] = dct
#
#dct = {}
#dct['log_z']   = False
#pdict['std']   = dct
#
#dct = {}
#dct['log_z']   = False
#pdict['sum']   = dct

def plot_nc(data_da,map_da,png_path,xlim=(0,24),ylim=None,**kwargs):

    stat    = data_da.attrs.get('stat')
    pdct    = pdict.get(stat,{})
    log_z   = pdct.get('log_z',True)

    freqs       = np.sort(data_da['freq_MHz'])[::-1]

    nx      = 100
    ny      = len(freqs)

    fig     = plt.figure(figsize=(30,4*ny))

    plt_nr  = 0
    for inx,freq in enumerate(freqs):
        plt_nr += 1
        ax = plt.subplot2grid((ny,nx),(inx,0),projection=ccrs.PlateCarree(),colspan=30)

        ax.coastlines(zorder=10,color='w')
        ax.plot(np.arange(10))
        map_data  = map_da.sel(freq_MHz=freq).copy()
        map_data  = np.log10(map_data)
        tf        = np.isneginf(map_data)
        map_data.values[tf] = 0
        map_data.plot.contourf(x=map_da.attrs['xkey'],y=map_da.attrs['ykey'],ax=ax,levels=30,cmap=mpl.cm.inferno)
        
        plt_nr += 1
        ax = plt.subplot2grid((ny,nx),(inx,35),colspan=65)
        data    = data_da.sel(freq_MHz=freq).copy()
        if log_z:
            tf          = data < 1.
            data        = np.log10(data)
            data.values[tf] = 0

#            data        = np.log10(data)
#            tf          = np.isneginf(data)
#            data.values[tf] = 0
            data.name   = 'log({})'.format(data.name)

        data.plot.contourf(x=data_da.attrs['xkey'],y=data_da.attrs['ykey'],ax=ax,levels=30)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

    fig.tight_layout()
    fig.savefig(png_path,bbox_inches='tight')
    plt.close(fig)

class ncLoader(object):
    def __init__(self,nc):
        with netCDF4.Dataset(nc) as nc_fl:
            groups  = [group for group in nc_fl.groups.keys()]

        das = OrderedDict()
        for group in groups:
            das[group] = OrderedDict()
            with xr.open_dataset(nc,group=group) as fl:
                ds      = fl.load()

            for param in ds.data_vars:
                das[group][param] = ds[param]

        xkeys   = groups.copy()
        xkeys.remove('map')

        self.nc     = nc
        self.das    = das
        self.xkeys  = xkeys

        self.get_map_da()

    def get_map_da(self):
        """
        Get map dataarray. This chooses the first variable in a map dataset.
        """
        map_das     = self.das['map']
        map_params  = [x for x in map_das]
        map_param   = map_params[0]
        map_da      = map_das[map_param]

        self.map_da     = map_da
        self.map_param  = map_param

def main(run_dct):
    srcs        = run_dct['srcs']
    baseout_dir = run_dct['baseout_dir']

    ncs = glob.glob(srcs)
    ncs.sort()

    for nc in ncs:
        ncl     = ncLoader(nc)
        map_da  = ncl.map_da
        bname   = os.path.basename(nc)[:-3]
        for xkey in ncl.xkeys:
            outdir  = os.path.join(baseout_dir,xkey)
            gl.prep_output({0:outdir})
            for param,data_da in ncl.das[xkey].items():
                fname   = '.'.join([bname,xkey,param,'png'])
                fpath   = os.path.join(outdir,fname)
                print(fpath)
                plot_nc(data_da,map_da,png_path=fpath,**run_dct)

if __name__ == '__main__':
    baseout_dir = 'output/galleries/histograms'

    run_dcts = []

    rd = {}
    this_dir            = 'sept2017'
    rd['srcs']          = 'data/histograms/{!s}/*.nc'.format('sept2017')
    rd['baseout_dir']   = os.path.join(baseout_dir,this_dir)
    run_dcts.append(rd)

    for rd in run_dcts:
        main(rd)
