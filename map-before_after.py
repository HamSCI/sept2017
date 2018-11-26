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

from util import geopack
from util.gen_lib import prep_output, BandData,fill_dark_side, band_legend
from util.plotutils import get_bins

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

de_prop         = {'marker':'^','edgecolor':'k','facecolor':'white'}
dxf_prop        = {'marker':'*','color':'blue'}
dxf_leg_size    = 150
dxf_plot_size   = 50

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

    sf  = .750  # Scale Factor
    fig = plt.figure(figsize=(sf*20,sf*8*ny))

    alphas  = ['a','b']
    for plt_inx,time_period in enumerate(time_periods):
        ax  = fig.add_subplot(ny,nx,plt_inx+1,projection=ccrs.PlateCarree())
        ax.coastlines()
        ax.gridlines()

        tf  = np.logical_and(df['occurred'] >= time_period[0],
                             df['occurred'] <  time_period[1])

        frame   = df[tf].copy()

        n_mids  = len(frame)

        xkey    = 'freq'
        cc      = frame[xkey]/1000.
        xx      = frame['md_long']
        yy      = frame['md_lat']

        plot_type   = 'path'
        
        for band,band_dct in BANDS.items():
            band_dct['count']   = np.sum(frame['band'] == band_dct['meters'])

        if plot_type == 'path':
            frame   = frame.sort_values('freq')
            for rinx,row in tqdm.tqdm(frame.iterrows(),total=len(frame)):
                xx  = [row['tx_long'],row['rx_long']]
                yy  = [row['tx_lat'], row['rx_lat']]
                clr = band_obj.get_rgba(row['freq']/1000)

                ax.plot(xx,yy,transform=ccrs.Geodetic(),color=clr)
                legend  = band_legend(ax,band_data=band_obj,rbn_rx=False)

            tx_df   = frame[['tx_long', 'tx_lat']].drop_duplicates()
#            label   = 'TX (N = {!s})'.format(len(tx_df))
#            tx_df.plot.scatter('tx_long', 'tx_lat', color="black", ax=ax, marker="o",label=label,zorder=20,s=1)

            rx_df   = frame[['rx_long', 'rx_lat']].drop_duplicates()
#            label   = 'RX (N = {!s})'.format(len(rx_df))
#            rx_df.plot.scatter('rx_long', 'rx_lat', ax=ax,zorder=30,s=10,**de_prop)

            text = []
            text.append('TX: {!s}'.format(len(tx_df)))
            text.append('RX: {!s}'.format(len(rx_df)))
            text.append('Paths: {!s}'.format(len(frame)))

            props = dict(facecolor='white', alpha=0.75,pad=6)
            ax.text(0.02,0.05,'\n'.join(text),transform=ax.transAxes,
                    ha='left',va='bottom',size='large',zorder=500,bbox=props)

        elif plot_type  == 'hist':
            bin_size    = 2.5
            lon_bins    = get_bins((-180,180),bin_size)
            lat_bins    = get_bins((-90,90),bin_size)

            hist, xb, yb = np.histogram2d(frame['md_long'],frame['md_lat'], bins=[lon_bins, lat_bins])

            cmap    = matplotlib.cm.viridis
            vmin    = 0
            vmax    = 100

            if vmin is None:
                vmin    = 0

            if vmax is None:
                vmax    = 0.8*np.max(hist)
                if np.sum(hist) == 0: vmax = 1.0

            levels  = np.linspace(vmin,vmax,30)

            norm    = matplotlib.colors.Normalize(vmin=vmin,vmax=vmax)
            pcoll   = ax.contourf(xb[:-1],yb[:-1],hist.T,levels,norm=norm,cmap=cmap)

            cbar    = plt.colorbar(pcoll,ax=ax)
            clabel  = 'Spot Density'
            fontdict = {'size':'xx-large','weight':'normal'}
            cbar.set_label(clabel,fontdict=fontdict)
        elif plot_type == 'scatter':
            cmap    = matplotlib.cm.jet
            vmin    = 0
            vmax    = 30

            label   = 'MidPt (N = {!s})'.format(n_mids)
            pcoll   = ax.scatter(xx,yy, c=cc, cmap=cmap, vmin=vmin, vmax=vmax, marker="o",label=label,zorder=10,s=10)

            cbar    = plt.colorbar(pcoll,ax=ax)
            cdct    = prmd[xkey]
            clabel  = cdct.get('label',xkey)
            fontdict = {'size':'xx-large','weight':'normal'}
            cbar.set_label(clabel,fontdict=fontdict)

#            tx_df   = frame[['tx_long', 'tx_lat']].drop_duplicates()
#            label   = 'TX (N = {!s})'.format(len(tx_df))
#            tx_df.plot.scatter('tx_long', 'tx_lat', color="black", ax=ax, marker="o",label=label,zorder=20,s=1)
#
#            rx_df   = frame[['rx_long', 'rx_lat']].drop_duplicates()
#            label   = 'RX (N = {!s})'.format(len(rx_df))
#            rx_df.plot.scatter('rx_long', 'rx_lat', color="blue", ax=ax, marker="*",label=label,zorder=30,s=10)
            ax.legend(loc='lower center',ncol=3)

        ax.set_xlim(regions[maplim_region]['lon_lim'])
        ax.set_ylim(regions[maplim_region]['lat_lim'])

        tp0_str = time_period[0].strftime('%Y %b %d %H%M UT')
        tp1_str = time_period[1].strftime('%Y %b %d %H%M UT')
        title   = '{!s} - {!s}'.format(tp0_str,tp1_str)
        ax.set_title(title)

        ax.set_title('({!s})'.format(alphas[plt_inx]),loc='left',fontdict={'size':24})
        mid_dt  = time_period[0] + (time_period[1]-time_period[0])/2
        fill_dark_side(ax,mid_dt,color='0.5',alpha=0.5)

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

