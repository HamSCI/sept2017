#!/usr/bin/python3
import os

from collections import OrderedDict

import matplotlib
matplotlib.use('Agg')

import numpy as np
import pandas as pd
import datetime
import cartopy.crs as ccrs
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import sys
import time
import tqdm

import geopack
from gen_lib import prep_output, BandData,fill_dark_side

sources     = OrderedDict()
sources[0]  = "dxcluster"
sources[1]  = "WSPRNet"
sources[2]  = "RBN"

rcp = matplotlib.rcParams
rcp['figure.titlesize']     = 'xx-large'
rcp['axes.titlesize']       = 'xx-large'
rcp['axes.labelsize']       = 'xx-large'
rcp['xtick.labelsize']      = 'xx-large'
rcp['ytick.labelsize']      = 'xx-large'
rcp['legend.fontsize']      = 'large'

rcp['figure.titleweight']   = 'bold'
rcp['axes.titleweight']     = 'bold'
rcp['axes.labelweight']     = 'bold'

# Parameter Dictionary
prmd = {}
tmp = {}
tmp['label']            = 'Solar Local Time [hr]'
prmd['slt_mid']         = tmp

tmp = {}
tmp['label']            = 'UT Hours'
prmd['ut_hrs']          = tmp

tmp = {}
tmp['label']            = 'f [MHz]'
prmd['freq']            = tmp

# Region Dictionary
regions = {}
tmp     = {}
tmp['lon_lim']  = (-180.,180.)
tmp['lat_lim']  = ( -90., 90.)
regions['World']    = tmp

tmp     = {}
tmp['lon_lim']  = (-130.,-60.)
tmp['lat_lim']  = (  20., 55.)
regions['US']   = tmp

tmp     = {}
tmp['lon_lim']  = ( -15., 55.)
tmp['lat_lim']  = (  30., 65.)
regions['Europe']   = tmp

tmp     = {}
tmp['lon_lim']  = ( -90.,-60.)
tmp['lat_lim']  = (  15., 30.)
regions['Carribean']    = tmp

tmp     = {}
tmp['lon_lim']  = ( -110.,-30.)
tmp['lat_lim']  = (    0., 45.)
regions['Greater Carribean']    = tmp

CSV_FILE_PATH   = "data/spot_csvs/{}.csv.bz2"
band_obj        = BandData()
BANDS           = band_obj.band_dict

def hours_from_dt64(dt64, date_):
    """ Take a datetime64 and return the value in decimal hours"""
    return (dt64 - date_).astype(float) / 3600

def plot_figure(time_periods,date_str,rgc_lim=None,maplim_region='World',filter_region=None,
        output_dir='output'):
    df = pd.read_csv(CSV_FILE_PATH.format(date_str))

    df["occurred"]  = pd.to_datetime(df["occurred"])
    df["ut_hrs"]    = hours_from_dt64(df["occurred"].values.astype("M8[s]"), np.datetime64(date_str))

    # Path Length Filtering
    if rgc_lim is not None:
        tf  = np.logical_and(df['dist_Km'] >= rgc_lim[0],
                             df['dist_Km'] <  rgc_lim[1])
        df  = df[tf].copy()

    cols = list(df) + ["md_lat", "md_long"]
    df = df.reindex(columns=cols)
    midpoints       = geopack.midpoint(df["tx_lat"], df["tx_long"], df["rx_lat"], df["rx_long"])
    df['md_lat']    = midpoints[0]
    df['md_long']   = midpoints[1]

    # Regional Filtering
    if filter_region is not None:
        df      = regional_filter(filter_region,df,kind=filter_region_kind)

    df['slt_mid']   = (df['ut_hrs'] + df['md_long']/15.) % 24.

    # Plotting #############################
    nx  = 1
    ny  = len(time_periods)
    nn  = 0

    sf  = 1.00  # Scale Factor
    fig = plt.figure(figsize=(sf*20,sf*8*ny))

    for plt_inx,time_period in enumerate(time_periods):
        ax  = fig.add_subplot(ny,nx,plt_inx+1,projection=ccrs.PlateCarree())
        ax.coastlines()
        ax.gridlines()

        tf  = np.logical_and(df['occurred'] >= time_period[0],
                             df['occurred'] <  time_period[1])

        frame   = df[tf].copy()

        n_mids  = len(frame)

        label   = 'MidPt (N = {!s})'.format(n_mids)

        cmap    = matplotlib.cm.jet
        vmin    = 0
        vmax    = 30

        xkey    = 'freq'
        cc      = frame[xkey]/1000.
        xx      = frame['md_long']
        yy      = frame['md_lat']

        for rinx,row in tqdm.tqdm(frame.iterrows(),total=len(frame)):
            xx  = [row['rx_lat'],row['rx_long']]
            yy  = [row['tx_lat'],row['tx_long']]

            ax.plot(xx,yy,transform=ccrs.Geodetic())

#        pcoll   = ax.scatter(xx,yy, c=cc, cmap=cmap, vmin=vmin, vmax=vmax, marker="o",label=label,zorder=10,s=10)
#        cbar    = plt.colorbar(pcoll,ax=ax)
#
#        cdct    = prmd[xkey]
#        clabel  = cdct.get('label',xkey)
#        fontdict = {'size':'xx-large','weight':'normal'}
#        cbar.set_label(clabel,fontdict=fontdict)
#
##        tx_df   = frame[['tx_long', 'tx_lat']].drop_duplicates()
##        label   = 'TX (N = {!s})'.format(len(tx_df))
##        tx_df.plot.scatter('tx_long', 'tx_lat', color="black", ax=ax, marker="o",label=label,zorder=20,s=1)
##
##        rx_df   = frame[['rx_long', 'rx_lat']].drop_duplicates()
##        label   = 'RX (N = {!s})'.format(len(rx_df))
##        rx_df.plot.scatter('rx_long', 'rx_lat', color="blue", ax=ax, marker="*",label=label,zorder=30,s=10)

        ax.set_xlim(regions[maplim_region]['lon_lim'])
        ax.set_ylim(regions[maplim_region]['lat_lim'])

        tp0_str = time_period[0].strftime('%Y %b %d %H%M UT')
        tp1_str = time_period[1].strftime('%Y %b %d %H%M UT')
        title   = '{!s} - {!s}'.format(tp0_str,tp1_str)
        ax.set_title(title)

        mid_dt  = time_period[0] + (time_period[1]-time_period[0])/2
        fill_dark_side(ax,mid_dt)

        ax.legend(loc='lower center',ncol=3)

    fig.tight_layout()

#    fdict   = {'size':50,'weight':'bold'}
#    fig.text(0.265,0.925,date_str,fontdict=fdict)

    fname   = date_str + ".png"
    fpath   = os.path.join(output_dir,fname)
    fig.savefig(fpath,bbox_inches='tight')
    plt.close(fig)

if __name__ == '__main__':
    output_dir  = 'output/galleries/maps'
    prep_output({0:output_dir},clear=True,php=True)

    flare_dt    = datetime.datetime(2017,9,6,11,53)
    window      = datetime.timedelta(minutes=15)
    date_str    = flare_dt.strftime('%Y-%m-%d')

    time_periods    = []
    time_periods.append( (flare_dt-window,flare_dt) )
    time_periods.append( (flare_dt,flare_dt+window) )

    plot_figure(time_periods,date_str,output_dir=output_dir)

