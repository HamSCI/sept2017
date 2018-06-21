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

layouts = {}

tmp = {}
#goes legend size = 15
layouts['default']  = tmp

tmp = {}
sf = 0.9
tmp['figsize']          = (sf*60,sf*30)
tmp['env_rspan']        = 2
tmp['band_rspan']       = 3
tmp['c0_cspan']         = 23
tmp['c1_pos']           = 25
tmp['c1_cspan']         = 75
tmp['nx']               = 100
tmp['map_cbar_shrink']  = 0.75
tmp['freq_size']        = 50
layouts['2band']        = tmp

tmp = {}
sf = 1.0
tmp['figsize']          = (sf*40,sf*30)
tmp['c0_cspan']         = 23
tmp['c1_pos']           = 27
tmp['c1_cspan']         = 73
tmp['nx']               = 100
tmp['title_size']       = 36
tmp['ticklabel_size']   = 24
tmp['label_size']       = 36
tmp['legend_size']      = 24
tmp['freq_size']        = 60
tmp['freq_xpos']        = -0.120
tmp['goes_lw']          = 5
tmp['kp_markersize']    = 20
layouts['4band12hr']    = tmp

def set_text_props(title_size='xx-large',ticklabel_size='xx-large',
        label_size='xx-large',legend_size='large',text_weight='bold',**kwargs):
    rcp = matplotlib.rcParams
    rcp['figure.titlesize']     = title_size 
    rcp['axes.titlesize']       = title_size 
    rcp['axes.labelsize']       = label_size
    rcp['xtick.labelsize']      = ticklabel_size 
    rcp['ytick.labelsize']      = ticklabel_size 
    rcp['legend.fontsize']      = legend_size

    rcp['figure.titleweight']   = text_weight
    rcp['axes.titleweight']     = text_weight
    rcp['axes.labelweight']     = text_weight

def make_histogram_from_dataframe(df: pd.DataFrame, ax: matplotlib.axes.Axes, title: str,
        xkey='occurred',xlim=None,ylim=(0,3000),vmin=None,vmax=None,log_hist=False,
        calc_hist_maxes=False,xlabels=True,plot_title=False):
    # TODO: Make all of this stuff configurable
    # Ultimately the goal is for this to be very versatile
    # x-axis: UTC

    xbin_0  = xlim[0].hour + xlim[0].minute/60. + xlim[0].second/3600.
    xbin_1  = xbin_0 + (xlim[1]-xlim[0]).total_seconds()/3600.
    xbins   = gl.get_bins((xbin_0,xbin_1), 10./60)

    # y-axis: distance (km)
    ybins = gl.get_bins(ylim, 500)

    tmp     = df[xkey] - df[xkey].min()
    total_hours = tmp.map(lambda x: x.total_seconds()/3600.)
    if len(df[xkey]) > 1:
        hist, xb, yb = np.histogram2d(total_hours, df["dist_Km"], bins=[xbins, ybins])
    else:
        xb      = xbins
        yb      = ybins
        hist    = np.zeros((len(xb)-1,len(yb)-1))

    vm_scale    = 0.6
    if log_hist:
        tf          = hist >= 1
        tmp         = np.log10(hist[tf])

        hist_1      = hist*0.
        hist_1[tf]  = tmp
        hist        = hist_1
        vm_scale    = 0.8

    if calc_hist_maxes:
        return hist

    if xlabels:
        xdct    = gl.prmd[xkey]
        xlabel  = xdct.get('label',xkey)
        ax.set_xlabel(xlabel)

    if plot_title:
        ax.set_title(title)

    ax.set_ylabel('R_gc [km]')

    # "borrowed" from SEQP
    if vmin is None:
        vmin    = 0

    if vmax is None:
        vmax    = vm_scale*np.max(hist)
        if np.sum(hist) == 0: vmax = 1.0

    levels  = np.linspace(vmin,vmax,15)

    cmap    = plt.cm.jet
    norm    = matplotlib.colors.Normalize(vmin=vmin,vmax=vmax)
    xb_dt   = [xlim[0]+datetime.timedelta(hours=(x-xbin_0)) for x in xb]
    pcoll   = ax.contourf(xb_dt[:-1],yb[:-1],hist.T,levels,norm=norm,cmap=cmap)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    cbar    = plt.colorbar(pcoll,ax=ax)
    if log_hist:
        cbar.set_label('log10(Ham Radio\nSpot Density)')
    else:
        cbar.set_label('Ham Radio\nSpot Density')

def make_figure(sTime,eTime,xkey='occurred',
        rgc_lim=(0,40000), maplim_region='World', filter_region=None, filter_region_kind='midpoints',
        log_hist=False,output_dir='output',calc_hist_maxes=False,fname=None,box=None,band_obj=None,
        cparam=None,layout=None):
    """
    xkey:   {'slt_mid','ut_hrs'}
    """

    if layout is None:
        lout = layouts.get('default')
    else:
        lout = layouts.get(layout)

    set_text_props(**lout)

    if band_obj is None:
        band_obj    = gl.BandData()

    band_dict   = band_obj.band_dict

    date_str_0  = sTime.strftime('%d %b %Y')
    date_str_1  = eTime.strftime('%d %b %Y')
    date_str    = '{!s} - {!s}'.format(date_str_0,date_str_1)

    print('Loading CSVs...')
    df      = pd.DataFrame()
    dates   = list(daterange(sTime, eTime+datetime.timedelta(hours=24)))
    if len(dates) == 0: dates = [sTime]
    for dt in tqdm.tqdm(dates):
        dft         = gl.load_spots_csv(dt.strftime("%Y-%m-%d"),rgc_lim=rgc_lim,
                        filter_region=filter_region,filter_region_kind=filter_region_kind)
        df          = df.append(dft,ignore_index=True)

    # Plotting #############################
    print('Plotting...')

    env_rspan   = lout.get('env_rspan',1)
    band_rspan  = lout.get('band_rspan',1)
    c0_span     = lout.get('c0_cspan',1)
    c1_span     = lout.get('c1_cspan',4)
    c1_pos      = lout.get('c1_pos',c0_span)

    n_env       = 2
    nx          = lout.get('nx',5)
    ny          = len(band_dict)*band_rspan + n_env*env_rspan
    
    sf      = lout.get('sf',1.00)  # Scale Factor
    figsize = lout.get('figsize',(sf*40, sf*4*len(band_dict)))

    fig = plt.figure(figsize=figsize)

    # Geospace Environment ####################
    axs_to_adjust   = []
    omni            = Omni()
    ax              = plt.subplot2grid((ny,nx),(0,c1_pos),colspan=c1_span,rowspan=env_rspan)
    msize           = lout.get('kp_markersize',10)
    dst_lw          = lout.get('goes_lw',2)
    omni_axs        = omni.plot_dst_kp(sTime,eTime,ax,xlabels=True,
                        kp_markersize=msize,dst_lw=dst_lw)
    axs_to_adjust   += omni_axs

    ########################################
    goes_dcts       = OrderedDict()
    goes_dcts[13]   = {}
    goes_dcts[15]   = {}

    for sat_nr,gd in goes_dcts.items():
        gd['data']      = goes.read_goes(sTime,eTime,sat_nr=sat_nr)
#        gd['flares']    = goes.find_flares(gd['data'],min_class='M5',window_minutes=60)
        gd['var_tags']  = ['B_AVG']
        gd['labels']    = ['GOES {!s}'.format(sat_nr)]

    ax              = plt.subplot2grid((ny,nx),(env_rspan,c1_pos),colspan=c1_span,rowspan=env_rspan)
    xdct            = gl.prmd[xkey]
    xlabel          = xdct.get('label',xkey)
    goes_lw         = lout.get('goes_lw',2)
    for sat_nr,gd in goes_dcts.items():
        goes.goes_plot(gd['data'],sTime,eTime,ax=ax,
                var_tags=gd['var_tags'],labels=gd['labels'],
                legendLoc='upper right',lw=goes_lw)

#    with open(os.path.join(output_dir,'{!s}-flares.txt'.format(date_str)),'w') as fl:
#        fl.write(flares.to_string())
#
#    for key,flare in flares.iterrows():
#        label   = '{0} Class Flare @ {1}'.format(flare['class'],key.strftime('%H%M UT'))
#        ut_hr   = goes.ut_hours(key)
#        ax.plot(ut_hr,flare['B_AVG'],'o',label=label,color='blue')
    ########################################

    title   = 'NOAA GOES X-Ray (0.1 - 0.8 nm) Irradiance'
    size    = lout.get('label_size',20)
    ax.text(0.01,0.05,title,transform=ax.transAxes,ha='left',fontdict={'size':size,'weight':'bold'})
    axs_to_adjust.append(ax)

    hist_maxes  = {}
    for band_inx, (band_key,band) in enumerate(band_dict.items()):
        fig_row = n_env*env_rspan + band_inx*band_rspan
        if fig_row == ny-1:
            xlabels = True
        else:
            xlabels = False

        frame   = df.loc[df["band"] == band.get('meters')].copy()
        frame.sort_values(xkey,inplace=True)

        n_mids  = len(frame)
        print('   {!s}: {!s} (n={!s})'.format(date_str,band.get('freq_name'),n_mids))

        # Histograms ########################### 
        ax      = plt.subplot2grid((ny,nx),(fig_row,c1_pos),
                    colspan=c1_span,rowspan=band_rspan)
        title   = '{!s} ({!s})'.format(date_str,band.get('freq_name'))

        vmin    = band.get('vmin')
        vmax    = band.get('vmax')

        hist    = make_histogram_from_dataframe(frame, ax, title,xkey=xkey,xlim=(sTime,eTime),ylim=rgc_lim,
                    vmin=vmin,vmax=vmax,calc_hist_maxes=calc_hist_maxes,xlabels=xlabels,log_hist=log_hist)

        fdict       = {'size':lout.get('freq_size',35),'weight':'bold'}
        freq_xpos   = lout.get('freq_xpos',-0.075)
        ax.text(freq_xpos,0.5,band.get('freq_name'),transform=ax.transAxes,va='center',rotation=90,fontdict=fdict)

        hist_ax = ax

        if calc_hist_maxes:
            hist_maxes[band_key]    = np.max(hist)
            continue
        
        #    # Map ################################## 
        ax = plt.subplot2grid((ny,nx),(fig_row,0),projection=ccrs.PlateCarree(),
                rowspan=band_rspan,colspan=c0_span)
        ax.coastlines()
        ax.gridlines()

        xx      = frame['md_long']
        yy      = frame['md_lat']
        if cparam is not None:
            cc      = frame[cparam]

        if len(xx) == 0:
            xx  = np.array([0,0])
            yy  = np.array([0,0])
            cc  = np.array([0,0])

        if cparam is None:
            pcoll   = ax.scatter(xx,yy,color='blue',marker="o",zorder=10,s=5)
        else:
            cdct    = gl.prmd[cparam]
            cmap    = cdct.get('cmap',matplotlib.cm.jet)
            vmin    = cdct.get('vmin',0)
            vmax    = cdct.get('vmax',np.nanmax(cc))

            pcoll   = ax.scatter(xx,yy, c=cc, cmap=cmap, vmin=vmin, vmax=vmax, marker="o",zorder=10,s=5)

            cb_props = {}
            cb_props['shrink']      = lout.get('map_cbar_shrink',1.)
            cb_props['fraction']    = lout.get('map_cbar_fraction',0.15)
            cb_props['pad']         = lout.get('map_cbar_pad',0.05)
            cb_props['aspect']      = lout.get('map_cbar_aspect',20)
            cbar    = plt.colorbar(pcoll,ax=ax,**cb_props)

            clabel  = cdct.get('label',cparam)
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

#    fdict   = {'size':50,'weight':'bold'}
#    title   = '{!s}-\n{!s}'.format(date_str_0,date_str_1)
#    fig.text(0.030,0.925,title,fontdict=fdict)

    xpos    = 0.030
    ypos    = 0.925
    fdict   = {'size':50,'weight':'bold'}
    title   = '{!s}-\n{!s}'.format(date_str_0,date_str_1)
    fig.text(xpos,ypos,title,fontdict=fdict)

    srcs    = '\n'.join([' '+x for x in gl.list_sources(df)])
    txt     = 'Ham Radio Networks\n' + srcs
    fdict   = {'size':30,'weight':'bold'}
    fig.text(xpos,ypos-0.080,txt,fontdict=fdict)

    if fname is None:
        fname   = '{!s}_{!s}_{!s}_map-{!s}_filter-{!s}-{!s}-PLOTTED{!s}.png'.format(
                date_str,xkey, rgc_lim, maplim_region, filter_region, filter_region_kind,
                datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')
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

    with mp.Pool() as pool:
        results = pool.map(plot_wrapper,run_dcts)

    for result in results:
        for band_key,band in band_dict.items():
            if 'hist_maxes' not in band.keys():
                band['hist_maxes'] = []
            band['hist_maxes'].append(result[band_key])

    for band_key,band in band_dict.items():
        band['vmax']    = np.percentile(band['hist_maxes'],85)

if __name__ == "__main__":
    output_dir  = 'output/galleries/summary-multiday'
    gl.prep_output({0:output_dir},clear=True)
    test_configuration  = True
    global_cbars        = False

    run_dcts    = []

    dct = {}
    dct['sTime']                = datetime.datetime(2017, 9, 6,6)
    dct['eTime']                = datetime.datetime(2017, 9, 6,18)
    dct['rgc_lim']              = (0,3000)
    dct['maplim_region']        = 'Europe'
    dct['filter_region']        =  dct['maplim_region']
    dct['filter_region_kind']   = 'mids'
    dct['band_obj']             = gl.BandData([7,14,21,28])
    dct['layout']               = '4band12hr'
    dct['output_dir']           = output_dir
    run_dcts.append(dct)

#    dct = {}
#    dct['sTime']                = datetime.datetime(2017, 9, 4)
#    dct['eTime']                = datetime.datetime(2017, 9, 14)
#    dct['rgc_lim']              = (0,3000)
#    dct['maplim_region']        = 'Europe'
#    dct['filter_region']        =  dct['maplim_region']
#    dct['filter_region_kind']   = 'mids'
#    dct['output_dir']           = output_dir
#    run_dcts.append(dct)
#
#    dct = {}
#    dct['sTime']                = datetime.datetime(2017, 9, 4)
#    dct['eTime']                = datetime.datetime(2017, 9, 14)
#    dct['rgc_lim']              = (0,3000)
#    dct['maplim_region']        = 'US'
#    dct['filter_region']        =  dct['maplim_region']
#    dct['filter_region_kind']   = 'mids'
#    dct['output_dir']           = output_dir
#    run_dcts.append(dct)
#
#    dct = {}
#    dct['sTime']                = datetime.datetime(2017, 9, 4)
#    dct['eTime']                = datetime.datetime(2017, 9, 14)
#    dct['rgc_lim']              = (0,3000)
#    dct['maplim_region']        = 'World'
#    dct['output_dir']           = output_dir
#    run_dcts.append(dct)
#
#    dct = {}
#    dct['sTime']                = datetime.datetime(2017, 9, 4)
#    dct['eTime']                = datetime.datetime(2017, 9, 14)
#    dct['rgc_lim']              = (0,20000)
#    dct['maplim_region']        = 'World'
#    dct['output_dir']           = output_dir
#    dct['log_hist']             = True
#    run_dcts.append(dct)
#
#    dct = {}
##    dct['sTime']                = datetime.datetime(2017, 9, 4)
##    dct['eTime']                = datetime.datetime(2017, 9, 14)
#    dct['sTime']                = datetime.datetime(2017, 9, 6)
#    dct['eTime']                = datetime.datetime(2017, 9, 10)
#    dct['rgc_lim']              = (0,10000)
#    dct['maplim_region']        = 'Greater Carribean'
#    dct['filter_region']        = 'Carribean'
#    dct['filter_region_kind']   = 'endpoints'
#    dct['log_hist']             = True
#    dct['band_obj']             = gl.BandData([7,14])
#    dct['cparam']               = 'dist_Km'
#    dct['layout']               = '2band'
#    dct['output_dir']           = output_dir
#    run_dcts.append(dct)

#    dct = dct.copy()
#    del dct['band_obj']
#    del dct['layout']
#    run_dcts.append(dct)


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
