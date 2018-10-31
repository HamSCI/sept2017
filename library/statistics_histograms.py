import os
import glob
import datetime
import dateutil
from collections import OrderedDict

import numpy as np
import pandas as pd
import xarray as xr

import tqdm

from . import gen_lib as gl

class KeyParamStore(object):
    def __init__(self,xkeys,params,prefix):
        self.xkeys      = xkeys
        self.params     = params
        self.prefix     = prefix
        self.data_das   = self._create_dict(xkeys,params)

    def _create_dict(self,xkeys,params):
        data_das = {}
        for xkey in xkeys:
            data_das[xkey] = {}
            for param in params:
                data_das[xkey][param] = [] 
        return data_das

    def load_nc(self,nc):
        for xkey in self.xkeys:
            group   = '{!s}/{!s}'.format(self.prefix,xkey)
            for param in self.params:
                with xr.open_dataset(nc,group=group) as fl:
                    ds = fl.load()
                self.data_das[xkey][param].append(ds[param])
        return self.data_das

    def concat(self,dim='ut_sTime'):
        for xkey in self.xkeys:
            for param in self.params:
                self.data_das[xkey][param] = xr.concat(self.data_das[xkey][param],dim=dim)
        return self.data_das

    def compute_stats(self,stats,dim='ut_sTime'):
        def pstat(param,stat):
            return '_'.join([param,stat])

        param_stats = []
        for param in self.params:
            for stat in stats:
                param_stats.append(pstat(param,stat))

        stats_dss   = self._create_dict(self.xkeys,[])

        for xkey in self.xkeys:
            stats_ds    = xr.Dataset()
            for param in self.params:
                for stat in stats:
                    data_da     = self.data_das[xkey][param]
                    stat_da     = eval("data_da.{!s}(dim='{!s}',keep_attrs=True)".format(stat,dim))
                    stat_da.attrs.update({'stat':stat})
                    stat_da.name    = pstat(param,stat)
                    stats_ds[pstat(param,stat)] = stat_da
            stats_dss[xkey] = stats_ds
        self.stats_dss = stats_dss
        return stats_dss

    def stats_to_nc(self,nc_path):
        for xkey,stats_ds in self.stats_dss.items():
            if os.path.exists(nc_path):
                mode = 'a'
            else:
                mode = 'w'

            group = '{!s}/{!s}'.format(self.prefix,xkey)
            stats_ds.to_netcdf(nc_path,mode=mode,group=group)

def main(run_dct):
    xkeys   = run_dct['xkeys']
    params  = run_dct['params']
    src_dir = run_dct['src_dir']
    stats   = run_dct['stats']

    # Set Up Data Storage Containers
    mps = KeyParamStore(xkeys,['spot_density'],'map')
    kps = KeyParamStore(xkeys,params,'time_series')

    ncs = glob.glob(os.path.join(src_dir,'*.data.nc'))
    ncs.sort()

    for nc in ncs:
        print(nc)
        mps.load_nc(nc)
        kps.load_nc(nc)

    mps.concat()
    kps.concat()

    mps.compute_stats(['sum'])
    kps.compute_stats(stats)

    stats_nc    = os.path.join(src_dir,'stats.nc')
    if os.path.exists(stats_nc):
        os.remove(stats_nc)

    mps.stats_to_nc(stats_nc)
    kps.stats_to_nc(stats_nc)

    print('Done computing statistics!!')
