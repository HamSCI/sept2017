#!/usr/bin/python3

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

from util import geopack
from util.plotutils import get_bins
from util.timeutils import daterange

CSV_FILE_PATH = "csvs/{}.csv.bz2"
BANDS = [160,80,40,20,15,10]

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


def make_histograms_by_date(date_str: str):
    df = pd.read_csv(CSV_FILE_PATH.format(date_str))
    df["occurred"] = pd.to_datetime(df["occurred"])
    df["occurred_hr"] = hours_from_dt64(df["occurred"].values.astype("M8[s]"), np.datetime64(date_str))
    fig = plt.figure(figsize=(7, 3.5*len(BANDS)))
    for i, band in enumerate(BANDS):
        frame = df.loc[df["band"] == band]
        ax = fig.add_subplot(len(BANDS), 1, i+1)
        title = date_str + " (" + str(band) + "m)"
        make_histogram_from_dataframe(frame, ax, title)
    fig.savefig("hist/" + date_str + ".png")

if __name__ == "__main__":
    sDate = datetime.datetime(2017, 9, 1)
    eDate = datetime.datetime(2017, 10, 1)
    for dt in daterange(sDate, eDate):
        print("Making histogram for", dt)
        date_str = dt.strftime("%Y-%m-%d")
        make_histograms_by_date(date_str)
