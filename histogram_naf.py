#!/usr/bin/python3

import os

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
from gen_lib import prep_output

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

CSV_FILE_PATH = "csvs/{}.csv.bz2"
BANDS = [160,80,40,20,15,10][::-1]


def get_bins(lim, bin_size):
    """ Helper function to split a limit into bins of the proper size """
    return np.arange(lim[0], lim[1]+2*bin_size, bin_size)

def hours_from_dt64(dt64, date_):
    """ Take a datetime64 and return the value in decimal hours"""
    return (dt64 - date_).astype(float) / 3600


def make_histogram_from_dataframe(df: pd.DataFrame, ax: matplotlib.axes.Axes, title: str):
    # TODO: Make all of this stuff configurable
    # Ultimately the goal is for this to be very versatile
    # x-axis: UTC
    xbins = get_bins((0, 24), 10./60)
    # y-axis: distance (km)
    ybins = get_bins((0, 3000), 500)


    # TODO: Clean this bit up, namely the hours_from_dt64 setup
    hist, xb, yb = np.histogram2d(df["occurred_hr"], df["dist_Km"], bins=[xbins, ybins])

    ax.set_title(title)

    # "borrowed" from SEQP
    vmin    = 0
    vmax    = 0.8*np.max(hist)
    if np.sum(hist) == 0:
        vmax = 1.0

    cmap    = plt.cm.jet
    pcoll   = ax.contourf(xb[:-1],yb[:-1],hist.T,15,vmin=vmin,vmax=vmax,cmap=cmap)
    cbar    = plt.colorbar(pcoll,ax=ax)
    cbar.set_label('Spot Density')


def make_histograms_by_date(date_str: str,output_dir='output'):
    df = pd.read_csv(CSV_FILE_PATH.format(date_str))
    df["occurred"] = pd.to_datetime(df["occurred"])
    df["occurred_hr"] = hours_from_dt64(df["occurred"].values.astype("M8[s]"), np.datetime64(date_str))

    cols = list(df) + ["md_lat", "md_long"]
    df = df.reindex(columns=cols)
    midpoints = geopack.midpoint(df["tx_lat"], df["tx_long"], df["rx_lat"], df["rx_long"])
    df['md_lat'] = midpoints[0]
    df['md_long'] = midpoints[1]


    fig = plt.figure(figsize=(30, 3.5*len(BANDS)))

    ax = fig.add_subplot(1, 2, 1, projection=ccrs.PlateCarree())
    ax.coastlines()
    ax.gridlines()

    tx_df = df[['tx_long', 'tx_lat']].drop_duplicates()
    print("Transmit_sites:", len(tx_df))
    print("Midpoints:", len(df))

    tx_patch = mpatches.Patch(color='red', label='Transmit Site (N=' + str(len(tx_df)) + ')')
    md_patch = mpatches.Patch(color='blue', label='Midpoint N=' + str(len(df)) + ')')

    df.plot.scatter('md_long', 'md_lat', color="blue", ax=ax, marker="o")
    tx_df.plot.scatter('tx_long', 'tx_lat', color="red", ax=ax, marker=".")

    ax.legend(handles=[tx_patch, md_patch],loc='lower left')
#    ax.set_title("Transmitter Sites (" + name + ")")

    nx  = 2
    ny  = len(BANDS)

    for row, band in enumerate(BANDS):
        frame   = df.loc[df["band"] == band]
#        ax      = fig.add_subplot(len(BANDS), 1, i+1)
        
        nn      = row*nx + 2
        ax      = fig.add_subplot(ny,nx,nn)
        title   = date_str + " (" + str(band) + "m)"
        make_histogram_from_dataframe(frame, ax, title)
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
