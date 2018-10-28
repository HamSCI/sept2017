#!/usr/bin/env python3
import os
import glob
import datetime
import dateutil
from collections import OrderedDict

import numpy as np
import pandas as pd
import xarray as xr
import netCDF4

import tqdm

from . import gen_lib as gl

class DataLoader(object):
    def __init__(self,nc,groups=None):
        if groups is None:
            groups  = self.get_groups(nc)

        dss   = OrderedDict()
        for group in groups:
            with xr.open_dataset(nc,group=group) as fl:
                ds = fl.load()
            dss[group] = ds

        self.dss    = dss
        self.groups = groups
        self.nc     = nc

    def get_groups(self,nc):
        with netCDF4.Dataset(nc) as nc_fl:
            groups  = [group for group in nc_fl.groups.keys()]
        return groups

class CompareBaseline(object):
    def __init__(self,nc,stats_obj,stats=None,groups=None,params=None):
        self.nc_in      = nc
        self.stats_obj  = stats_obj
        self.dl         = DataLoader(nc,groups)

        self._init_outfile()
        for group in groups:
            self.calc_statistic(stats,group,params)

    def _init_outfile(self):
        self.nc_out  = self.nc_in[:-8]+'.baseline_compare.nc'
        with xr.open_dataset(self.nc_in,group='map') as fl:
            ds  = fl.load()
        ds.to_netcdf(self.nc_out,mode='w',group='map')

    def calc_statistic(self,stats,group,params=None):
        ds      = self.dl.dss[group]
        if params is None:
            params  = [x for x in ds.data_vars]

        ds_out  = xr.Dataset()
        for param in params:
            for stat in stats:
                da      = ds[param]
                mean    = self.stats_obj.dss[group][pstat(param,'mean')]
                std     = self.stats_obj.dss[group][pstat(param,'std')]

                if stat == 'pct_err':
                    result          = (da - mean)/mean
                elif stat == 'z_score':
                    result          = (da - mean)/std

                attrs           = da.attrs.copy()
                attrs['stat']   = stat
                result.attrs    = attrs
                name            = pstat(param,stat)
                ds_out[name]    = result
        ds_out.to_netcdf(self.nc_out,mode='a',group=group)

def pstat(param,stat):
    return '_'.join([param,stat])

def main(run_dct):
    src_dir     = run_dct['src_dir']
    xkeys       = run_dct['xkeys']
    params      = run_dct.get('params')
    stats       = run_dct['stats']
    stats_nc    = run_dct.get('stats_nc')

    if stats_nc is None:
        stats_nc    = os.path.join(src_dir,'stats.nc')
    stats_obj   = DataLoader(stats_nc)

    ncs = glob.glob(os.path.join(src_dir,'*.data.nc'))
    ncs.sort()

    for nc in ncs:
        print(nc)
        cmp_obj = CompareBaseline(nc,stats_obj,stats=stats,groups=xkeys,params=params)

if __name__ == '__main__':
    run_dcts = []

    rd = {}
    rd['src_dir']   = 'data/histograms/active'
    rd['xkeys']     = ['ut_hrs','slt_mid']
    rd['stats']     = ['pct_err','z_score']

    run_dcts.append(rd)

    for rd in run_dcts:
        main(rd)
