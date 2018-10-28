#!/usr/bin/python3
import os
import glob
import datetime
import dateutil
from collections import OrderedDict
import ast

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
from timeutils import daterange

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

dct = {}
dct['log_z']        = False
pdict['z_score']    = dct

dct = {}
dct['log_z']        = False
pdict['pct_err']    = dct


class ncLoader(object):
    def __init__(self,sTime,eTime=None,srcs=None,**kwargs):
        if eTime is None:
            eTime = sTime + datetime.timedelta(hours=24)

        self.sTime      = sTime
        self.eTime      = eTime
        self.srcs       = srcs

        self._set_basename()
        self._get_fnames()
        self._load_ncs()

    def _set_basename(self):
        bname   = os.path.basename(self.srcs)
        bname   = bname.strip('*.')
        bname   = bname.strip('.nc')
        self.basename   = bname

    def _get_fnames(self):
        tmp = glob.glob(self.srcs)
        tmp.sort()
        
        dates       = daterange(self.sTime,self.eTime)
        date_strs   = [x.strftime('%Y%m%d') for x in dates]

        fnames      = []
        for fn in tmp:
            bn  = os.path.basename(fn)
            if bn[:8] in date_strs:
                fnames.append(fn)

        self.fnames = fnames
        return fnames
            
    def _load_ncs(self):
        dss     = OrderedDict()
        maps    = []

        for nc in self.fnames:
            # Identify Groups in netCDF File
            with netCDF4.Dataset(nc) as nc_fl:
                groups  = [group for group in nc_fl.groups.keys()]

            # Store DataSets (dss) from each group in an OrderedDict()
            for group in groups:
                with xr.open_dataset(nc,group=group) as fl:
                    ds      = fl.load()

                if group == 'map':
                    maps.append(ds)
                else:
                    # Calculate time vector relative to self.sTime
                    hrs         = np.array(ds.coords[group])
                    dt_0        = pd.Timestamp(np.array(ds['ut_sTime']).tolist())
                    time_vec    = [(dt_0 + pd.Timedelta(hours=x) - self.sTime).total_seconds()/3600. for x in hrs]

                    ds.coords[group]        = time_vec
                    ds.coords['ut_sTime']   = [self.sTime]

                    if group not in dss:
                        dss[group]  = []
                    dss[group].append(ds)

        # Concatenate and sum maps
        maps        = xr.concat(maps,'ut_sTime')
        maps        = maps.sum('ut_sTime',keep_attrs=True)
        self.maps   = maps

        # Concatenate other groups. 

        xlim        = (0, (self.eTime-self.sTime).total_seconds()/3600.)
        for group,ds_list in dss.items():
            ds          = xr.concat(ds_list,group)
            for data_var in ds.data_vars:
                attrs   = ds[data_var].attrs
                attrs.update({'xlim':str(xlim)})
                ds[data_var].attrs = attrs
            dss[group]  = ds

        self.datasets   = dss

    def plot(self,baseout_dir='output',xlim=None,ylim=None,xunits='datetime',**kwargs):
        map_da  = self.maps['spot_density']
        xlim_in = xlim

        for group,ds in self.datasets.items():
            outdir  = os.path.join(baseout_dir,group)
            gl.prep_output({0:outdir},clear=False)

            for data_var in ds.data_vars:
                data_da = ds[data_var].copy()

                xlim    = xlim_in
                if xlim is None:
                    xlim = ast.literal_eval(data_da.attrs.get('xlim','None'))

                if ylim is None:
                    ylim = ast.literal_eval(data_da.attrs.get('ylim','None'))

                if xunits == 'datetime':
                    hrs         = np.array(data_da.coords[group])
                    dt_vec      = [self.sTime + pd.Timedelta(hours=x) for x in hrs]
                    data_da.coords[group] = dt_vec

                    if xlim is not None:
                        xlim_0      = pd.Timedelta(hours=xlim[0]) + self.sTime
                        xlim_1      = pd.Timedelta(hours=xlim[1]) + self.sTime
                        xlim        = (xlim_0,xlim_1)

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
                    tf        = map_data < 1
                    map_data  = np.log10(map_data)
                    map_data.values[tf] = 0
                    map_data.name   = 'log({})'.format(map_data.name)
                    map_data.plot.contourf(x=map_da.attrs['xkey'],y=map_da.attrs['ykey'],ax=ax,levels=30,cmap=mpl.cm.inferno)
                    
                    plt_nr += 1
                    ax = plt.subplot2grid((ny,nx),(inx,35),colspan=65)
                    data    = data_da.sel(freq_MHz=freq).copy()
                    if log_z:
                        tf          = data < 1.
                        data        = np.log10(data)
                        data.values[tf] = 0
                        data.name   = 'log({})'.format(data.name)

                    data.plot.contourf(x=data_da.attrs['xkey'],y=data_da.attrs['ykey'],ax=ax,levels=30)

                    ax.set_xlim(xlim)
                    ax.set_ylim(ylim)

                fig.tight_layout()

                sTime_str   = self.sTime.strftime('%Y%m%d.%H%MUT')
                eTime_str   = self.eTime.strftime('%Y%m%d.%H%MUT')
                date_str    = '-'.join([sTime_str,eTime_str])

                fname   = '.'.join([date_str,self.basename,group,data_var,'png'])
                fpath   = os.path.join(outdir,fname)
                fig.savefig(fpath,bbox_inches='tight')
                plt.close(fig)

def main(run_dct):
    nc_obj      = ncLoader(**run_dct)
    nc_obj.plot(**run_dct)

if __name__ == '__main__':
    baseout_dir = 'output/galleries/histograms'

    run_dcts = []

    rd = {}
    this_dir            = 'sept2017'
#    rd['srcs']          = 'data/histograms/{!s}/*.baseline_compare.nc'.format(this_dir)
    rd['srcs']          = 'data/histograms/{!s}/*.data.nc'.format(this_dir)
    rd['baseout_dir']   = os.path.join(baseout_dir,this_dir)
    rd['sTime']         = datetime.datetime(2017,9,1)
    rd['eTime']         = datetime.datetime(2017,9,3)
    run_dcts.append(rd)

    for rd in run_dcts:
        main(rd)