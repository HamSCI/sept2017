#!/usr/bin/python3
import os
import sys
import time
import datetime
import multiprocessing as mp
from collections import OrderedDict

import matplotlib
matplotlib.use('Agg')

import numpy as np
import pandas as pd
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

import tqdm

import geopack
from timeutils import daterange
import goes
from omni import Omni

import gen_lib as gl

band_obj        = gl.BandData()
BANDS           = band_obj.band_dict

def make_histogram_from_dataframe(df: pd.DataFrame, ax: matplotlib.axes.Axes, title: str,
        xkey='ut_hrs',ylim=(0,3000),vmin=None,vmax=None,log_hist=False,
        calc_hist_maxes=False,xlabels=True,plot_title=False):
    # TODO: Make all of this stuff configurable
    # Ultimately the goal is for this to be very versatile
    # x-axis: UTC
    xbins = gl.get_bins((0, 24), 10./60)
    # y-axis: distance (km)
    ybins = gl.get_bins(ylim, 500)

    if len(df[xkey]) > 1:
        hist, xb, yb = np.histogram2d(df[xkey], df["dist_Km"], bins=[xbins, ybins])
    else:
        xb      = xbins
        yb      = ybins
        hist    = np.zeros((len(xb)-1,len(yb)-1))

    if log_hist:
        tf          = hist >= 1
        tmp         = np.log10(hist[tf])

        hist_1      = hist*0.
        hist_1[tf]  = tmp
        hist        = hist_1

    if calc_hist_maxes:
        return hist

    if xlabels:
        xdct    = gl.prmd[xkey]
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
    if log_hist:
        cbar.set_label('log10(Ham Radio\nSpot Density)')
    else:
        cbar.set_label('Ham Radio\nSpot Density')

def make_figure(date_str: str,xkey='ut_hrs',
        rgc_lim=(0,40000), maplim_region='World', filter_region=None, filter_region_kind='midpoints',
        log_hist=False,output_dir='output',calc_hist_maxes=False,fname=None,box=None):
    """
    xkey:   {'slt_mid','ut_hrs'}
    """

    print('Loading {!s}...'.format(date_str))
    df      = gl.load_spots_csv(date_str,rgc_lim=rgc_lim,
                    filter_region=filter_region,filter_region_kind=filter_region_kind)

    # Plotting #############################
    print('Plotting {!s}...'.format(date_str))
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
    omni_axs        = omni.plot_dst_kp(sDate,eDate,ax,xkey='ut_hrs',xlabels=True)
#    ax.set_xlabel('UT Hours')
    axs_to_adjust   += omni_axs


    ########################################
    goes_dcts       = OrderedDict()
    goes_dcts[13]   = {}
    goes_dcts[15]   = {}

    for sat_nr,gd in goes_dcts.items():
        gd['data']      = goes.read_goes(sDate,sat_nr=sat_nr)
#        gd['flares']    = goes.find_flares(gd['data'],min_class='M5',window_minutes=60)
        gd['var_tags']  = ['B_AVG']
        gd['labels']    = ['GOES {!s}'.format(sat_nr)]

    nn              += 2
    ax              = fig.add_subplot(ny,nx,nn)
    xdct            = gl.prmd[xkey]
    xlabel          = xdct.get('label',xkey)
    for sat_nr,gd in goes_dcts.items():
        goes.goes_plot_hr(gd['data'],ax,
                var_tags=gd['var_tags'],labels=gd['labels'],
                xkey=xkey,legendLoc='upper right',legendSize=15,lw=2)

#    with open(os.path.join(output_dir,'{!s}-flares.txt'.format(date_str)),'w') as fl:
#        fl.write(flares.to_string())
#
#    for key,flare in flares.iterrows():
#        label   = '{0} Class Flare @ {1}'.format(flare['class'],key.strftime('%H%M UT'))
#        ut_hr   = goes.ut_hours(key)
#        ax.plot(ut_hr,flare['B_AVG'],'o',label=label,color='blue')
    ########################################

#    ax.set_xlabel(xlabel)
    title   = 'NOAA GOES X-Ray (0.1 - 0.8 nm) Irradiance'
    ax.text(0.02,0.05,title,transform=ax.transAxes,ha='left',fontdict={'size':20,'weight':'bold'})
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
                    vmin=vmin,vmax=vmax,calc_hist_maxes=calc_hist_maxes,xlabels=xlabels,log_hist=log_hist)

        fdict   = {'size':35,'weight':'bold'}
        ax.text(-0.1725,0.5,band.get('freq_name'),transform=ax.transAxes,va='center',rotation=90,fontdict=fdict)

        hist_ax = ax

        if calc_hist_maxes:
            hist_maxes[band_key]    = np.max(hist)
            continue
        
        #    # Map ################################## 
        nn      = fig_row*nx + 1

        ax = fig.add_subplot(ny,nx,nn, projection=ccrs.PlateCarree())
        ax.coastlines()
        ax.gridlines()

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

        pcoll   = ax.scatter(xx,yy, c=cc, cmap=cmap, vmin=vmin, vmax=vmax, marker="o",zorder=10,s=5)
        cbar    = plt.colorbar(pcoll,ax=ax)

        cdct    = gl.prmd[xkey]
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

        if box is not None:
            rgn = gl.regions.get(box)
            x0  = rgn['lon_lim'][0]
            y0  = rgn['lat_lim'][0]
            ww  = rgn['lon_lim'][1] - x0
            hh  = rgn['lat_lim'][1] - y0
            
            p   = matplotlib.patches.Rectangle((x0,y0),ww,hh,fill=False,zorder=500)
            ax.add_patch(p)

        ax.set_xlim(gl.regions[maplim_region]['lon_lim'])
        ax.set_ylim(gl.regions[maplim_region]['lat_lim'])

        label   = 'Midpoints (N = {!s})'.format(n_mids)
        fdict   = {'size':24}
        ax.text(0.5,-0.15,label,transform=ax.transAxes,ha='center',fontdict=fdict)

    if calc_hist_maxes:
        plt.close(fig)
        return hist_maxes

    fig.tight_layout()

    # Force geospace environment axes to line up with histogram
    # axes even though it doesn't have a color bar.
    for ax_0 in axs_to_adjust:
        gl.adjust_axes(ax_0,hist_ax)

    xpos    = 0.230
    ypos    = 0.965
    fdict   = {'size':50,'weight':'bold'}
    fig.text(xpos,ypos,date_str,fontdict=fdict)

    srcs    = '\n'.join([' '+x for x in gl.list_sources(df)])
    txt     = 'Ham Radio Networks\n' + srcs
    fdict   = {'size':30,'weight':'bold'}
    fig.text(xpos,ypos-0.065,txt,fontdict=fdict)

    if fname is None:
        fname   = '{!s}_{!s}_{!s}_map-{!s}_filter-{!s}-{!s}.png'.format(
                date_str,xkey, rgc_lim, maplim_region, filter_region, filter_region_kind
                )
    fpath   = os.path.join(output_dir,fname)
    fig.savefig(fpath,bbox_inches='tight')
    plt.close(fig)

def plot_wrapper(run_dct):
    result  = make_figure(**run_dct)
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
    output_dir  = 'output/galleries/summary-daily'
    gl.prep_output({0:output_dir},clear=True)
    test_configuration  = True
    global_cbars        = False

    run_dcts    = []

    dct = {}
    dct['date_str']             = '2017-09-06'
    dct['xkey']                 = 'ut_hrs'
    dct['rgc_lim']              = (0,3000)
    dct['maplim_region']        = 'Europe'
    dct['filter_region']        =  dct['maplim_region']
    dct['filter_region_kind']   = 'mids'
    dct['output_dir']           = output_dir
    dct['fname']                = '2017-09-06-EU.png'
    run_dcts.append(dct)

    dct = {}
    dct['date_str']             = '2017-09-06'
    dct['xkey']                 = 'ut_hrs'
    dct['rgc_lim']              = (0,3000)
    dct['maplim_region']        = 'US'
    dct['filter_region']        =  dct['maplim_region']
    dct['filter_region_kind']   = 'mids'
    dct['output_dir']           = output_dir
    dct['fname']                = '2017-09-06-US.png'
    run_dcts.append(dct)

    dct = {}
    dct['date_str']             = '2017-09-08'
    dct['xkey']                 = 'ut_hrs'
    dct['rgc_lim']              = (0,3000)
    dct['maplim_region']        = 'World'
    dct['filter_region']        = None
    dct['filter_region_kind']   = 'mids'
    dct['output_dir']           = output_dir
    dct['fname']                = '2017-09-08.png'
    run_dcts.append(dct)

    dct = {}
    dct['date_str']             = '2017-09-08'
    dct['xkey']                 = 'ut_hrs'
    dct['rgc_lim']              = (0,20000)
    dct['maplim_region']        = 'World'
    dct['output_dir']           = output_dir
    dct['log_hist']             = True
    run_dcts.append(dct)

#    dct = {}
#    dct['date_str']             = '2017-09-08'
#    dct['xkey']                 = 'ut_hrs'
#    dct['rgc_lim']              = (0,20000)
#    dct['maplim_region']        = 'Greater Carribean'
#    dct['filter_region']        = 'Carribean'
#    dct['filter_region_kind']   = 'endpoints'
#    dct['output_dir']           = output_dir
#    dct['fname']                = '2017-09-08.png'
#    run_dcts.append(dct)

#    sDate = datetime.datetime(2017, 9, 1)
#    eDate = datetime.datetime(2017, 9, 3)
##    eDate = datetime.datetime(2017, 10, 1)
#
#    run_dcts    = []
#    for dt in daterange(sDate, eDate):
#        dct = {}
#        dct['date_str']     = dt.strftime("%Y-%m-%d")
#        dct['output_dir']   = output_dir
#        run_dcts.append(dct)

    if test_configuration:
        print('Plotting...')
        for run_dct in run_dcts:
            plot_wrapper(run_dct)
    else:
        if global_cbars:
            print('Calculating Limits...')
            calculate_limits(run_dcts)

        print('Plotting...')
        with mp.Pool() as pool:
            results = pool.map(plot_wrapper,run_dcts)
