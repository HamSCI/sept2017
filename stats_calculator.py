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

def histogram_from_df(df,xb_size_min=10.,yb_size_km=250.,xlim=None,ylim=(0,40000),xkey='occurred'):
    ret_dct = {}

    if xlim is None:
        xlim    = (df[xkey].min(),df[xkey].max())

    xbin_0  = xlim[0].hour + xlim[0].minute/60. + xlim[0].second/3600.
    xbin_1  = xbin_0 + (xlim[1]-xlim[0]).total_seconds()/3600.
    xbins   = gl.get_bins((xbin_0,xbin_1), xb_size_min/60)

    # y-axis: distance (km)
    ybins = gl.get_bins(ylim, yb_size_km)

    dt_min      = df[xkey].min()
    df_dt_0     = datetime.datetime(dt_min.year,dt_min.month,dt_min.day)
    tmp         = df[xkey] - df_dt_0
    total_hours = tmp.map(lambda x: x.total_seconds()/3600.)
    if len(df[xkey]) > 2:
       hist, xb, yb = np.histogram2d(total_hours, df["dist_Km"], bins=[xbins, ybins])
    else:
        xb      = xbins
        yb      = ybins
        hist    = np.zeros((len(xb)-1,len(yb)-1))

    coords  = {'sTime':[sTime],'ut_hours':xbins[:-1],'rgc_km':ybins[:-1]}
    dims    = ('sTime','ut_hours','rgc_km')
    values  = np.expand_dims(hist,0)
    hist_xr = xr.DataArray(values,coords,dims)
    return hist_xr

def gen_csv(df,output_dir,fname='df_out',gen_callIds=False):
    csv_keys    = OrderedDict()
#    csv_keys['tx_loc_source']   = 'tx_loc_source'
#    csv_keys['rx_loc_source']   = 'rx_loc_source'
#    csv_keys['rpt_key']         = 'rpt_key'
#    csv_keys['rpt_mode']        = 'rpt_mode'
#    csv_keys['band']            = 'band'
#    csv_keys['tx_grid']         = 'tx_grid'
#    csv_keys['rx_grid']         = 'rx_grid'
#    csv_keys['tx_mode']         = 'mode'
    csv_keys['occurred']        = 'datetime'
    csv_keys['freq']            = 'frequency_MHz'
    csv_keys['source']          = 'source'
    csv_keys['dist_Km']         = 'dist_Km'
    csv_keys['snr']             = 'snr_dB'
    csv_keys['tx']              = 'tx'
    csv_keys['tx_lat']          = 'tx_lat'
    csv_keys['tx_long']         = 'tx_long'
    csv_keys['rx']              = 'rx'
    csv_keys['rx_lat']          = 'rx_lat'
    csv_keys['rx_long']         = 'rx_long'
    csv_keys['md_lat']          = 'md_lat'
    csv_keys['md_long']         = 'md_long'
#    csv_keys['ut_hrs']          = 'ut_hrs'
    csv_keys['slt_mid']         = 'slt_mid_hrs'
    keys    = [x for x in csv_keys.keys()]

    df_out  = df[keys].copy()
    df_out  = df_out.rename(columns=csv_keys)
    df_out['frequency_MHz']     = df_out['frequency_MHz']/1000.

    for src_id,src_dct  in gl.sources.items():
        tf  = df_out['source'] == src_id
        df_out.loc[tf,'source'] = src_dct['name']

    # Associate Call Signs #################
    call_csv    = 'data/callIds/callsigns.csv.bz2'
    if os.path.exists(call_csv):
        print('Associating call signs...')
        call_df     = pd.read_csv(call_csv,compression='bz2').set_index('call_ids')
        for txrx in ['tx','rx']:
            df_out          = df_out.join(call_df,on=txrx)
            df_out[txrx]    = df_out['call']
            del df_out['call']

    csv_name    = '{!s}.csv.bz2'.format(fname)
    fpath       = os.path.join(output_dir,csv_name)
    df_out.to_csv(fpath,index=False,compression='bz2')

    call_ids    = df['tx'].tolist() + df['rx'].tolist()
    call_ids    = pd.DataFrame({'call_ids':call_ids})
    call_ids.sort_values('call_ids',inplace=True)
    call_ids.drop_duplicates(inplace=True)
    
    if gen_callIds:
        csv_name    = '{!s}_callIds.csv.bz2'.format(fname)
        fpath       = os.path.join(output_dir,csv_name)
        call_ids.to_csv(fpath,index=False,compression='bz2')

class SpotDay(object):
    def __init__(self,sTime,rgc_lim=(0,40000), 
            filter_region=None,filter_region_kind='midpoints',
            xb_size_min=10.,yb_size_km=250.):

        attr        = OrderedDict()
        self.attr   = attr
        attr['sTime']               = sTime
        attr['rgc_lim']             = rgc_lim
        attr['filter_region']       = filter_region
        attr['filter_region_kind']  = filter_region_kind
        attr['xb_size_min']         = xb_size_min
        attr['yb_size_km']          = yb_size_km

        df      = gl.load_spots_csv(sTime.strftime("%Y-%m-%d"),rgc_lim=rgc_lim,
                    filter_region=filter_region,filter_region_kind=filter_region_kind)
        self.df = df

        band_obj    = gl.BandData()
        band_dict   = band_obj.band_dict
        self.calculate_histograms(band_dict)

    def calculate_histograms(self,band_dict):
        df          = self.df
        xb_size_min = self.attr.get('xb_size_min')
        yb_size_km  = self.attr.get('yb_size_km')
        sTime       = self.attr.get('sTime')
        ylim        = self.attr.get('rgc_lim')
        xlim        = (sTime,sTime+datetime.timedelta(hours=24))

        hists       = []
        for band_inx, (band_key,band) in enumerate(band_dict.items()):
            frame   = df.loc[df["band"] == band.get('meters')].copy()

            dct                 = {}
            dct['df']           = frame
            dct['xb_size_min']  = xb_size_min
            dct['yb_size_km']   = yb_size_km
            dct['xlim']         = xlim
            dct['ylim']         = ylim

            hist    = histogram_from_df(**dct)
            hist    = hist.assign_coords(band=int(band_key))
            hists.append(hist)

        self.hists  = xr.concat(hists,dim='band')

if __name__ == "__main__":
    sDate       = datetime.datetime(2017,  9, 1)
    eDate       = datetime.datetime(2017,  9, 1)
    dates       = daterange(sDate,eDate)

    run_dcts    = []
    for date in dates:
        dct = {}
        dct['sTime']                = date
        run_dcts.append(dct)

    spot_days   = OrderedDict()
    for run_dct in run_dcts:
        sTime               = run_dct.get('sTime')
        sd                  = SpotDay(**run_dct)
        spot_days[sTime]    = sd

    import ipdb; ipdb.set_trace()
