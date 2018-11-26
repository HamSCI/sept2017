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

import multiprocessing as mp

from util import geopack
from util.gen_lib import prep_output, BandData
from util.plotutils import get_bins
from util.timeutils import daterange

import goes
from omni import Omni

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

def adjust_axes(ax_0,ax_1):
    """
    Force geospace environment axes to line up with histogram
    axes even though it doesn't have a color bar.
    """
    ax_0_pos    = list(ax_0.get_position().bounds)
    ax_1_pos    = list(ax_1.get_position().bounds)
    ax_0_pos[2] = ax_1_pos[2]
    ax_0.set_position(ax_0_pos)

def regional_filter(region,df,kind='mids'):
    rgnd    = regions[region]
    lat_lim = rgnd['lat_lim']
    lon_lim = rgnd['lon_lim']

    if kind == 'mids':
        tf_md_lat   = np.logical_and(df.md_lat >= lat_lim[0], df.md_lat < lat_lim[1])
        tf_md_long  = np.logical_and(df.md_long >= lon_lim[0], df.md_long < lon_lim[1])
        tf_0        = np.logical_and(tf_md_lat,tf_md_long)
        tf          = tf_0
        df          = df[tf].copy()
    elif kind == 'endpoints':
        tf_rx_lat   = np.logical_and(df.rx_lat >= lat_lim[0], df.rx_lat < lat_lim[1])
        tf_rx_long  = np.logical_and(df.rx_long >= lon_lim[0], df.rx_long < lon_lim[1])
        tf_rx       = np.logical_and(tf_rx_lat,tf_rx_long)

        tf_tx_lat   = np.logical_and(df.tx_lat >= lat_lim[0], df.tx_lat < lat_lim[1])
        tf_tx_long  = np.logical_and(df.tx_long >= lon_lim[0], df.tx_long < lon_lim[1])
        tf_tx       = np.logical_and(tf_tx_lat,tf_tx_long)
        tf          = np.logical_or(tf_rx,tf_tx)

        df          = df[tf].copy()

    return df

def make_histogram_from_dataframe(df: pd.DataFrame, ax: matplotlib.axes.Axes, title: str,
        xkey='ut_hrs',ylim=(0,3000),vmin=None,vmax=None,calc_hist_maxes=False,xlabels=True,plot_title=False):
    # TODO: Make all of this stuff configurable
    # Ultimately the goal is for this to be very versatile
    # x-axis: UTC
    xbins = get_bins((0, 24), 10./60)
    # y-axis: distance (km)
    ybins = get_bins(ylim, 500)

    # TODO: Clean this bit up, namely the hours_from_dt64 setup
    if len(df[xkey]) > 1:
        hist, xb, yb = np.histogram2d(df[xkey], df["dist_Km"], bins=[xbins, ybins])
    else:
        xb      = xbins
        yb      = ybins
        hist    = np.zeros((len(xb)-1,len(yb)-1))

    if calc_hist_maxes:
        return hist

    if xlabels:
        xdct    = prmd[xkey]
        xlabel  = xdct.get('label',xkey)
        ax.set_xlabel(xlabel)
#    else:
#        for xtl in ax.get_xticklabels():
#            xtl.set_visible(False)

    if plot_title:
        ax.set_title(title)

    ax.set_ylabel('R_gc [km]')

    # "borrowed" from SEQP
    if vmin is None:
        vmin    = 0

    if vmax is None:
        vmax    = 0.8*np.max(hist)
        if np.sum(hist) == 0: vmax = 1.0

    levels  = np.linspace(vmin,vmax,15)

    cmap    = plt.cm.jet
    norm    = matplotlib.colors.Normalize(vmin=vmin,vmax=vmax)
    pcoll   = ax.contourf(xb[:-1],yb[:-1],hist.T,levels,norm=norm,cmap=cmap)
    ax.set_ylim(ylim)
    cbar    = plt.colorbar(pcoll,ax=ax)
    cbar.set_label('Spot Density')

def make_histograms_by_date(date_str: str,xkey='ut_hrs',output_dir='output',calc_hist_maxes=False):
    """
    xkey:   {'slt_mid','ut_hrs'}
    """
    rgc_lim             = (0, 3000)
#    rgc_lim             = (0, 10000)
#    maplim_region       = 'World'
    maplim_region       = 'Greater Carribean'
    filter_region       = 'Carribean'
    filter_region_kind  = 'endpoints'

    df = pd.read_csv(CSV_FILE_PATH.format(date_str))

    df["occurred"]  = pd.to_datetime(df["occurred"])
    df["ut_hrs"]    = hours_from_dt64(df["occurred"].values.astype("M8[s]"), np.datetime64(date_str))

    # Path Length Filtering
    tf  = np.logical_and(df['dist_Km'] >= rgc_lim[0],
                         df['dist_Km'] <  rgc_lim[1])
    df  = df[tf].copy()

    cols = list(df) + ["md_lat", "md_long"]
    df = df.reindex(columns=cols)
    midpoints       = geopack.midpoint(df["tx_lat"], df["tx_long"], df["rx_lat"], df["rx_long"])
    df['md_lat']    = midpoints[0]
    df['md_long']   = midpoints[1]

    # Regional Filtering
    df      = regional_filter(filter_region,df,kind=filter_region_kind)

    df['slt_mid']   = (df['ut_hrs'] + df['md_long']/15.) % 24.

    # Plotting #############################

    nx  = 2
    ny  = len(BANDS)+2
    nn  = 0

    sf  = 1.00  # Scale Factor
    fig = plt.figure(figsize=(sf*30, sf*4*len(BANDS)))

    # Geospace Environment ####################
    axs_to_adjust   = []
    sDate           = datetime.datetime.strptime(date_str,'%Y-%m-%d')
    eDate           = sDate + datetime.timedelta(days=1)

    nn              += 2
    omni            = Omni()
    ax              = fig.add_subplot(ny,nx,nn)
    omni_axs        = omni.plot_dst_kp(sDate,eDate,ax,xlabels=False)
    axs_to_adjust   += omni_axs


    ########################################
    goes_data       = goes.read_goes(sDate)
#    goes_data       = goes.read_goes(sDate,sat_nr=13)
    flares          = goes.find_flares(goes_data,min_class='M5',window_minutes=60)

    nn              += 2
    ax              = fig.add_subplot(ny,nx,nn)
    xdct            = prmd[xkey]
    xlabel          = xdct.get('label',xkey)
    goes.goes_plot_hr(goes_data,ax,xkey=xkey)

    with open(os.path.join(output_dir,'{!s}-flares.txt'.format(date_str)),'w') as fl:
        fl.write(flares.to_string())

    for key,flare in flares.iterrows():
        label   = '{0} Class Flare @ {1}'.format(flare['class'],key.strftime('%H%M UT'))
        ut_hr   = goes.ut_hours(key)
        ax.plot(ut_hr,flare['B_AVG'],'o',label=label,color='blue')
    ########################################

    ax.set_xlabel(xlabel)
    axs_to_adjust.append(ax)

    hist_maxes  = {}
    for fig_row, (band_key,band) in enumerate(BANDS.items()):
        fig_row += ny-len(BANDS)
        if fig_row == ny-1:
            xlabels = True
        else:
            xlabels = False

        frame   = df.loc[df["band"] == band.get('meters')].copy()
        frame.sort_values(xkey,inplace=True)

        n_mids  = len(frame)
        print('   {!s}: {!s} (n={!s})'.format(date_str,band.get('freq_name'),n_mids))

        # Histograms ########################### 
        nn      = fig_row*nx + 2
        ax      = fig.add_subplot(ny,nx,nn)
        title   = '{!s} ({!s})'.format(date_str,band.get('freq_name'))

        vmin    = band.get('vmin')
        vmax    = band.get('vmax')

        hist    = make_histogram_from_dataframe(frame, ax, title,xkey=xkey,ylim=rgc_lim,
                    vmin=vmin,vmax=vmax,calc_hist_maxes=calc_hist_maxes,xlabels=xlabels)

        fname   = band.get('freq_name')
        fdict   = {'size':35,'weight':'bold'}
        ax.text(-0.1725,0.5,fname,transform=ax.transAxes,va='center',rotation=90,fontdict=fdict)

        hist_ax = ax

        if calc_hist_maxes:
#            if 'hist_maxes' not in band.keys():
#                band['hist_maxes'] = []
#            band['hist_maxes'].append(np.max(hist))
            hist_maxes[band_key]    = np.max(hist)
            continue
        
        #    # Map ################################## 
        nn      = fig_row*nx + 1

        ax = fig.add_subplot(ny,nx,nn, projection=ccrs.PlateCarree())
        ax.coastlines()
        ax.gridlines()

        label   = 'MidPt (N = {!s})'.format(n_mids)
#        frame.plot.scatter('md_long', 'md_lat', color=color, ax=ax, marker="o",label=label,zorder=10,s=10)

        cmap    = matplotlib.cm.jet
        vmin    = 0
        vmax    = 24

        cc      = frame[xkey]
        xx      = frame['md_long']
        yy      = frame['md_lat']

        if len(xx) == 0:
            xx  = np.array([0,0])
            yy  = np.array([0,0])
            cc  = np.array([0,0])

        pcoll   = ax.scatter(xx,yy, c=cc, cmap=cmap, vmin=vmin, vmax=vmax, marker="o",label=label,zorder=10,s=10)
        cbar    = plt.colorbar(pcoll,ax=ax)

        cdct    = prmd[xkey]
        clabel  = cdct.get('label',xkey)
        fontdict = {'size':'xx-large','weight':'normal'}
        cbar.set_label(clabel,fontdict=fontdict)

#        tx_df   = frame[['tx_long', 'tx_lat']].drop_duplicates()
#        label   = 'TX (N = {!s})'.format(len(tx_df))
#        tx_df.plot.scatter('tx_long', 'tx_lat', color="black", ax=ax, marker="o",label=label,zorder=20,s=1)
#
#        rx_df   = frame[['rx_long', 'rx_lat']].drop_duplicates()
#        label   = 'RX (N = {!s})'.format(len(rx_df))
#        rx_df.plot.scatter('rx_long', 'rx_lat', color="blue", ax=ax, marker="*",label=label,zorder=30,s=10)

        ax.set_xlim(regions[maplim_region]['lon_lim'])
        ax.set_ylim(regions[maplim_region]['lat_lim'])

        ax.legend(loc='lower center',ncol=3)

    if calc_hist_maxes:
        plt.close(fig)
        return hist_maxes

    fig.tight_layout()

    # Force geospace environment axes to line up with histogram
    # axes even though it doesn't have a color bar.
    for ax_0 in axs_to_adjust:
        adjust_axes(ax_0,hist_ax)

    fdict   = {'size':50,'weight':'bold'}
    fig.text(0.265,0.925,date_str,fontdict=fdict)

    fname   = date_str + ".png"
    fpath   = os.path.join(output_dir,fname)
    fig.savefig(fpath,bbox_inches='tight')
    plt.close(fig)

def plot_wrapper(run_dct):
    result  = make_histograms_by_date(**run_dct)
    return result

def calculate_limits(run_dcts):
    """
    Finds best spot density colorbar limits for each band given all plots
    in the set.
    """

    this_rdcts = []
    for run_dct in run_dcts:
        tmp = run_dct.copy()
        tmp['calc_hist_maxes'] = True
        this_rdcts.append(tmp)
    run_dcts    = this_rdcts

#    results = []
#    for run_dct in run_dcts:
#        result  = plot_wrapper(run_dct)
#        results.append(result)

    with mp.Pool() as pool:
        results = pool.map(plot_wrapper,run_dcts)

    for result in results:
        for band_key,band in BANDS.items():
            if 'hist_maxes' not in band.keys():
                band['hist_maxes'] = []
            band['hist_maxes'].append(result[band_key])

    for band_key,band in BANDS.items():
        band['vmax']    = np.percentile(band['hist_maxes'],85)

if __name__ == "__main__":
    output_dir  = 'output/galleries/hist'
    prep_output({0:output_dir},clear=True,php=True)
    test_configuration  = False

    sDate = datetime.datetime(2017, 9, 1)
    eDate = datetime.datetime(2017, 9, 16)
#    eDate = datetime.datetime(2017, 10, 1)

    run_dcts    = []
    for dt in daterange(sDate, eDate):
        dct = {}
        dct['date_str']     = dt.strftime("%Y-%m-%d")
        dct['output_dir']   = output_dir
        run_dcts.append(dct)

    if test_configuration:
        print('Plotting...')
        for run_dct in run_dcts:
            plot_wrapper(run_dct)
    else:
        print('Calculating Limits...')
        calculate_limits(run_dcts)

        print('Plotting...')
        with mp.Pool() as pool:
            results = pool.map(plot_wrapper,run_dcts)
