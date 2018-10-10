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
    """
    xkey:   {'slt_mid','ut_hrs'}
    """

    xkey        = attrs['xkey']
    xb_size_min = attrs['xb_size_min']
    yb_size_km  = attrs['yb_size_km']
    rgc_lim     = attrs['rgc_lim']

    # x-axis: time (hours)
    xbin_0  = 0.
    xbin_1  = 24.
    xbins   = gl.get_bins((xbin_0,xbin_1), xb_size_min/60.)

    # y-axis: distance (km)
    ybins   = gl.get_bins(rgc_lim, yb_size_km)
    hrs     = frame[xkey]

    if len(frame[xkey]) > 2:
       hist, xb, yb = np.histogram2d(hrs, frame["dist_Km"], bins=[xbins, ybins])
    else:
        xb      = xbins
        yb      = ybins
        hist    = np.zeros((len(xb)-1,len(yb)-1))

    crds    = []
#   crds.append( (name, vals) )
    crds.append( ('ut_sTime', [attrs['sTime']]) )
    crds.append( ('freq_MHz', [attrs['band']])  )
    crds.append( (xkey, xb[:-1]) )
    crds.append( ('rgc_km',yb[:-1]) )
   
    data    = hist
    data    = np.expand_dims(data,0)
    data    = np.expand_dims(data,0)
    
    for key,val in attrs.items():
        attrs[key] = str(val)
    hist_xr = xr.DataArray(data,crds,attrs=attrs)
    return hist_xr

def main(run_dct):
    # Get Variables from run_dct
    sDate               = run_dct['sDate']
    eDate               = run_dct['eDate']
    param               = run_dct['param']
    rgc_lim             = run_dct['rgc_lim']
    filter_region       = run_dct['filter_region']
    filter_region_kind  = run_dct['filter_region_kind']
    band_obj            = run_dct['band_obj']
    xkey                = run_dct['xkey']
    xb_size_min         = run_dct['xb_size_min']
    yb_size_km          = run_dct['yb_size_km']
    output_dir          = run_dct['output_dir']

    # Define path for saving NetCDF Files
    tmp = []
    tmp.append('{!s}'.format(param))
    tmp.append('{!s}'.format(xkey))
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

        # Cycle through bands
        hist_xrs = []
        for band_inx, (band_key,band) in enumerate(band_obj.band_dict.items()):
            frame   = df.loc[df["band"] == band.get('meters')].copy()

            # Create attrs diction to save with xarray DataArray
            attrs  = OrderedDict()
            attrs['sTime']                 = dt
            attrs['param']                 = param
            attrs['xkey']                  = xkey
            attrs['rgc_lim']               = rgc_lim
            attrs['filter_region']         = filter_region
            attrs['filter_region_kind']    = filter_region_kind
            attrs['xb_size_min']           = xb_size_min
            attrs['yb_size_km']            = yb_size_km
            attrs['band']                  = band_key
            attrs['band_name']             = band['name']
            attrs['band_fname']            = band['freq_name']

            # Compute Histogram
            result  = calc_histogram(frame,attrs)
            hist_xrs.append(result)

        # Concatenate all bands into single DataArray
        hist_xr = xr.concat(hist_xrs,dim='freq_MHz')

        # Create NC Name
        tmp = []
        tmp.append(dt.strftime('%Y%m%d'))
        nc_name = '_'.join(tmp) + '.nc'
        nc_path = os.path.join(ncs_path,nc_name)
        hist_xr.to_netcdf(nc_path)

if __name__ == "__main__":
    output_dir  = 'data/histograms'
    gl.prep_output({0:output_dir},clear=False)

    run_dcts    = []

    rd  = {}
    rd['sDate']                 = datetime.datetime(2017,9,1)
    rd['eDate']                 = datetime.datetime(2017,9,2)
    rd['param']                 = 'spot_density'
    rd['xkey']                  = 'ut_hrs'
    rd['rgc_lim']               = (0,40000)
    rd['filter_region']         = None
    rd['filter_region_kind']    = 'midpoints'
    rd['xb_size_min']           = 30.
    rd['yb_size_km']            = 500.
    rd['output_dir']            = output_dir
    rd['band_obj']              = gl.BandData()
    run_dcts.append(rd)

    for rd in run_dcts:
        main(rd)
