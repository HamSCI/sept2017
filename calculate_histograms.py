#!/usr/bin/python3
import os
import datetime
from collections import OrderedDict

import numpy as np
import pandas as pd
import xarray as xr

import tqdm

from timeutils import daterange
import gen_lib as gl

def calc_histogram(frame,attrs):
    xkey    = attrs['xkey']
    xlim    = attrs['xlim']
    dx      = attrs['dx']
    ykey    = attrs['ykey']
    ylim    = attrs['ylim']
    dy      = attrs['dy']

    xbins   = gl.get_bins(xlim,dx)
    ybins   = gl.get_bins(ylim,dy)

    if len(frame) > 2:
       hist, xb, yb = np.histogram2d(frame[xkey], frame[ykey], bins=[xbins, ybins])
    else:
        xb      = xbins
        yb      = ybins
        hist    = np.zeros((len(xb)-1,len(yb)-1))

    crds    = {}
    crds['ut_sTime']    = attrs['sTime']
    crds['freq_MHz']    = attrs['band']
    crds[xkey]          = xb[:-1]
    crds[ykey]          = yb[:-1]
    
    attrs   = attrs.copy()
    for key,val in attrs.items():
        attrs[key] = str(val)
    da = xr.DataArray(hist,crds,attrs=attrs,dims=[xkey,ykey])
    return da 

def main(run_dct):
    # Get Variables from run_dct
    sDate               = run_dct['sDate']
    eDate               = run_dct['eDate']
    params              = run_dct['params']
    xkeys               = run_dct['xkeys']
    rgc_lim             = run_dct['rgc_lim']
    filter_region       = run_dct['filter_region']
    filter_region_kind  = run_dct['filter_region_kind']
    band_obj            = run_dct['band_obj']
    xb_size_min         = run_dct['xb_size_min']
    yb_size_km          = run_dct['yb_size_km']
    output_dir          = run_dct['output_dir']

    # Define path for saving NetCDF Files
    tmp = []
    if filter_region is not None:
        tmp.append('{!s}'.format(filter_region))
        tmp.append('{!s}'.format(filter_region_kind))
    tmp.append('{:.0f}-{:.0f}km'.format(rgc_lim[0],rgc_lim[1]))
    tmp.append('dx{:.0f}min'.format(xb_size_min))
    tmp.append('dy{:.0f}km'.format(yb_size_km))
    ncs_path = os.path.join(output_dir,'_'.join(tmp))
    gl.prep_output({0:ncs_path},clear=False)

    # Loop through dates
    dates   = list(daterange(sDate, eDate))[:-1]
    for dt in tqdm.tqdm(dates):
        # Load spots from CSVs
        df         = gl.load_spots_csv(dt.strftime("%Y-%m-%d"),rgc_lim=rgc_lim,
                        filter_region=filter_region,filter_region_kind=filter_region_kind)

        # Set Up Data Storage Containers
        data_das = {}
        for xkey in xkeys:
            data_das[xkey] = {}
            for param in params:
                data_das[xkey][param] = []

        map_hist_das    = []

        # Cycle through bands
        for band_inx, (band_key,band) in enumerate(band_obj.band_dict.items()):
            frame   = df.loc[df["band"] == band.get('meters')].copy()

            # Create attrs diction to save with xarray DataArray
            attrs  = OrderedDict()
            attrs['sTime']              = dt
            attrs['param']              = param
            attrs['filter_region']      = filter_region
            attrs['filter_region_kind'] = filter_region_kind
            attrs['band']               = band_key
            attrs['band_name']          = band['name']
            attrs['band_fname']         = band['freq_name']

            # Compute Map
            attrs['xkey']               = 'md_long'
            attrs['xlim']               = (-180,180)
            attrs['dx']                 = 1
            attrs['ykey']               = 'md_lat'
            attrs['ylim']               = (-90,90)
            attrs['dy']                 = 1
            result  = calc_histogram(frame,attrs)
            map_hist_das.append(result)

            for xkey in xkeys:
                for param in params:
                    # Compute General Data
                    attrs['xkey']               = xkey
                    attrs['param']              = param
                    attrs['xlim']               = (0,24)
                    attrs['dx']                 = xb_size_min/60.
                    attrs['ykey']               = 'dist_Km'
                    attrs['ylim']               = rgc_lim
                    attrs['dy']                 = yb_size_km
                    result  = calc_histogram(frame,attrs)
                    data_das[xkey][param].append(result)

        # Maps - Concatenate all bands into single DataArray
        map_hist_da     = xr.concat(map_hist_das,dim='freq_MHz')

        map_ds          = xr.Dataset()
        map_ds['spot_density']  = map_hist_da
       
        # Time Series - Concatenate all bands into single DataArray
        for xkey in xkeys:
            for param in params:
                data_das[xkey][param] = xr.concat(data_das[xkey][param],dim='freq_MHz')

        # Data Sets
        data_dss    = OrderedDict()
        for xkey in xkeys:
            data_ds = xr.Dataset()
            for param in params:
                data_ds[param] = data_das[xkey][param]
            data_dss[xkey] = data_ds

        # Save to data file.
        nc_name = dt.strftime('%Y%m%d') + '.data.nc'
        nc_path = os.path.join(ncs_path,nc_name)
        map_ds.to_netcdf(nc_path,mode='w',group='/map')
        for xkey,data_ds in data_dss.items():
            group = '/{!s}'.format(xkey)
            data_ds.to_netcdf(nc_path,mode='a',group=group)

if __name__ == "__main__":
    output_dir  = 'data/histograms'
    gl.prep_output({0:output_dir},clear=False)

    run_dcts    = []

#    rd  = {}
#    rd['sDate']                 = datetime.datetime(2016,1,1)
#    rd['eDate']                 = datetime.datetime(2018,1,1)
#    rd['params']                = ['spot_density']
#    rd['xkeys']                 = ['ut_hrs','slt_mid']
#    rd['rgc_lim']               = (0,10000)
#    rd['filter_region']         = None
#    rd['filter_region_kind']    = 'mids'
#    rd['xb_size_min']           = 30.
#    rd['yb_size_km']            = 500.
#    rd['output_dir']            = output_dir
#    rd['band_obj']              = gl.BandData()
#    run_dcts.append(rd)

#    rd  = {}
#    rd['sDate']                 = datetime.datetime(2017,9,1)
#    rd['eDate']                 = datetime.datetime(2017,10,1)
#    rd['params']                = ['spot_density']
#    rd['xkeys']                 = ['ut_hrs','slt_mid']
#    rd['rgc_lim']               = (0,10000)
#    rd['filter_region']         = 'US'
#    rd['filter_region_kind']    = 'mids'
#    rd['xb_size_min']           = 30.
#    rd['yb_size_km']            = 500.
#    rd['output_dir']            = output_dir
#    rd['band_obj']              = gl.BandData()
#    run_dcts.append(rd)

    rd  = {}
    rd['sDate']                 = datetime.datetime(2017,9,1)
    rd['eDate']                 = datetime.datetime(2017,10,1)
    rd['params']                = ['spot_density']
    rd['xkeys']                 = ['ut_hrs','slt_mid']
    rd['rgc_lim']               = (0,10000)
    rd['filter_region']         = 'Europe'
    rd['filter_region_kind']    = 'mids'
    rd['xb_size_min']           = 30.
    rd['yb_size_km']            = 500.
    rd['output_dir']            = output_dir
    rd['band_obj']              = gl.BandData()
    run_dcts.append(rd)

    for rd in run_dcts:
        main(rd)
