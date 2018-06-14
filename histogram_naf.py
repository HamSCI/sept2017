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

import geopack
from timeutils import daterange
from gen_lib import prep_output, BandData


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

CSV_FILE_PATH   = "csvs/{}.csv.bz2"
band_obj        = BandData()
BANDS           = band_obj.band_dict

def get_bins(lim, bin_size):
    """ Helper function to split a limit into bins of the proper size """
    return np.arange(lim[0], lim[1]+2*bin_size, bin_size)

def hours_from_dt64(dt64, date_):
    """ Take a datetime64 and return the value in decimal hours"""
    return (dt64 - date_).astype(float) / 3600


def make_histogram_from_dataframe(df: pd.DataFrame, ax: matplotlib.axes.Axes, title: str,
        xkey='slt_mid',ylim=(0,3000)):
    # TODO: Make all of this stuff configurable
    # Ultimately the goal is for this to be very versatile
    # x-axis: UTC
    xbins = get_bins((0, 24), 10./60)
    # y-axis: distance (km)
    ybins = get_bins(ylim, 500)


    # TODO: Clean this bit up, namely the hours_from_dt64 setup
    hist, xb, yb = np.histogram2d(df[xkey], df["dist_Km"], bins=[xbins, ybins])

    xdct    = prmd[xkey]
    xlabel  = xdct.get('label',xkey)
    ax.set_xlabel(xlabel)
    ax.set_title(title)

    # "borrowed" from SEQP
    vmin    = 0
    vmax    = 0.8*np.max(hist)
    if np.sum(hist) == 0:
        vmax = 1.0

    cmap    = plt.cm.jet
    pcoll   = ax.contourf(xb[:-1],yb[:-1],hist.T,15,vmin=vmin,vmax=vmax,cmap=cmap)
    ax.set_ylim(ylim)
    cbar    = plt.colorbar(pcoll,ax=ax)
    cbar.set_label('Spot Density')


def make_histograms_by_date(date_str: str,xkey='slt_mid',output_dir='output'):
    """
    xkey:   {'slt_mid','ut_hrs'}
    """
    df = pd.read_csv(CSV_FILE_PATH.format(date_str))

    df["occurred"]  = pd.to_datetime(df["occurred"])
    df["ut_hrs"]    = hours_from_dt64(df["occurred"].values.astype("M8[s]"), np.datetime64(date_str))

    # Filtering
    rgc_lim = (0, 3000)
    tf  = np.logical_and(df['dist_Km'] >= rgc_lim[0],
                         df['dist_Km'] <  rgc_lim[1])
    df  = df[tf].copy()

    cols = list(df) + ["md_lat", "md_long"]
    df = df.reindex(columns=cols)
    midpoints       = geopack.midpoint(df["tx_lat"], df["tx_long"], df["rx_lat"], df["rx_long"])
    df['md_lat']    = midpoints[0]
    df['md_long']   = midpoints[1]

    df['slt_mid']   = (df['ut_hrs'] + df['md_long']/15.) % 24.

    nx  = 2
    ny  = len(BANDS)

    sf  = 1.00  # Scale Factor
    fig = plt.figure(figsize=(sf*30, sf*4*len(BANDS)))

    for fig_row, band in enumerate(BANDS.values()):
        frame   = df.loc[df["band"] == band.get('meters')]
        frame.sort_values(xkey,inplace=True)
        
        #    # Map ################################## 
        nn      = fig_row*nx + 1

        ax = fig.add_subplot(ny,nx,nn, projection=ccrs.PlateCarree())
        ax.coastlines()
        ax.gridlines()
        n_mids  = len(frame)
        color   = band.get('color')
        print('{!s} (n={!s})'.format(band.get('freq_name'),n_mids))

        label   = 'MidPt (N = {!s})'.format(n_mids)
#        frame.plot.scatter('md_long', 'md_lat', color=color, ax=ax, marker="o",label=label,zorder=10,s=10)

        cmap    = matplotlib.cm.jet
        vmin    = 0
        vmax    = 24

        c       = frame[xkey]
        xx      = frame['md_long']
        yy      = frame['md_lat']
        pcoll   = ax.scatter(xx,yy, c=c, cmap=cmap, vmin=vmin, vmax=vmax,
                    marker="o",label=label,zorder=10,s=10)
        cbar    = plt.colorbar(pcoll,ax=ax)

        cdct    = prmd[xkey]
        clabel  = cdct.get('label',xkey)
        fontdict = {'size':'xx-large','weight':'normal'}
        cbar.set_label(clabel,fontdict=fontdict)

#        tx_df   = frame[['tx_long', 'tx_lat']].drop_duplicates()
#        label   = 'TX (N = {!s})'.format(len(tx_df))
#        tx_df.plot.scatter('tx_long', 'tx_lat', color="black", ax=ax, marker="o",label=label,zorder=20,s=1)

#        rx_df   = frame[['rx_long', 'rx_lat']].drop_duplicates()
#        label   = 'RX (N = {!s})'.format(len(rx_df))
#        rx_df.plot.scatter('rx_long', 'rx_lat', color="blue", ax=ax, marker="*",label=label,zorder=30,s=10)

        ax.set_xlim(-180,180)
        ax.set_ylim(-90,90)
        ax.legend(loc='lower center',ncol=3)

        # Histograms ########################### 
        nn      = fig_row*nx + 2
        ax      = fig.add_subplot(ny,nx,nn)
        title   = '{!s} ({!s})'.format(date_str,band.get('freq_name'))
        make_histogram_from_dataframe(frame, ax, title,xkey=xkey,ylim=rgc_lim)

    fig.tight_layout()

    fname   = date_str + ".png"
    fpath   = os.path.join(output_dir,fname)
    fig.savefig(fpath,bbox_inches='tight')
    plt.close(fig)

if __name__ == "__main__":
    output_dir  = 'output/hist'
    prep_output({0:output_dir},clear=True)

    sDate = datetime.datetime(2017, 9, 1)
    eDate = datetime.datetime(2017, 10, 1)
    for dt in daterange(sDate, eDate):
        print("Making histogram for", dt)
        date_str = dt.strftime("%Y-%m-%d")
        make_histograms_by_date(date_str,output_dir=output_dir)
