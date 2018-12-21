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

from library import geopack
from library import gen_lib as gl

de_prop         = {'marker':'^','edgecolor':'k','facecolor':'white'}
dxf_prop        = {'marker':'*','color':'blue'}
dxf_leg_size    = 150
dxf_plot_size   = 50

band_obj        = gl.BandData(cb_safe=True)
BANDS           = band_obj.band_dict

def plot_figure(time_periods,date_str,rgc_lim=None,maplim_region='World',
        filter_region=None,filter_region_kind='mids',output_dir='output'):

    print('Loading {!s}...'.format(date_str))
    df      = gl.load_spots_csv(date_str,rgc_lim=rgc_lim,
                    filter_region=filter_region,filter_region_kind=filter_region_kind)

    tf = np.logical_and(df['freq'] >= 1800, df['freq'] <= 30000)
    df = df[tf].copy()

    # Plotting #############################
    print('Plotting {!s}...'.format(date_str))
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
#        frame   = frame[:1000].copy()

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
            for band,band_dct in BANDS.items():
                tf  = frame['band'] == band_dct['meters']
                frame_band = frame[tf].copy()
                clr = band_dct['color']

                for rinx,row in tqdm.tqdm(frame_band.iterrows(),total=len(frame_band)):
                    xx  = [row['tx_long'],row['rx_long']]
                    yy  = [row['tx_lat'], row['rx_lat']]
#                    clr = band_obj.get_rgba(row['freq']/1000)

                    ax.plot(xx,yy,transform=ccrs.Geodetic(),color=clr,zorder=band)

            legend  = gl.band_legend(ax,band_data=band_obj,rbn_rx=False)

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
            lon_bins    = gl.get_bins((-180,180),bin_size)
            lat_bins    = gl.get_bins((-90,90),bin_size)

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
            cdct    = gl.prmd[xkey]
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

        ax.set_xlim(gl.regions[maplim_region]['lon_lim'])
        ax.set_ylim(gl.regions[maplim_region]['lat_lim'])

        tp0_str = time_period[0].strftime('%Y %b %d %H%M UT')
        tp1_str = time_period[1].strftime('%Y %b %d %H%M UT')
        title   = '{!s} - {!s}'.format(tp0_str,tp1_str)
        ax.set_title(title)

        ax.set_title('({!s})'.format(alphas[plt_inx]),loc='left',fontdict={'size':24})
        mid_dt  = time_period[0] + (time_period[1]-time_period[0])/2
        gl.fill_dark_side(ax,mid_dt,color='0.5',alpha=0.5)

    fig.tight_layout()

#    fdict   = {'size':50,'weight':'bold'}
#    fig.text(0.265,0.925,date_str,fontdict=fdict)

    fname   = date_str + ".png"
    fpath   = os.path.join(output_dir,fname)
    fig.savefig(fpath,bbox_inches='tight')
    plt.close(fig)

if __name__ == '__main__':
    output_dir  = 'output/galleries/maps'
    gl.prep_output({0:output_dir},clear=True,php=True)

    flare_dt    = datetime.datetime(2017,9,6,11,53)
    window      = datetime.timedelta(minutes=15)
    date_str    = flare_dt.strftime('%Y-%m-%d')

    time_periods    = []
    time_periods.append( (flare_dt-window,flare_dt) )
    time_periods.append( (flare_dt,flare_dt+window) )

    plot_figure(time_periods,date_str,output_dir=output_dir)

