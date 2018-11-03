#!/usr/bin/python3
import os
import sys
import time
import datetime
import multiprocessing as mp
from collections import OrderedDict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import PolyCollection
import cartopy.crs as ccrs

import numpy as np
import pandas as pd

import tqdm

from library import geopack
from library.timeutils import daterange
from library import goes
from library.omni import Omni

from library import gen_lib as gl

layouts = {}

tmp = {}
#goes legend size = 15
layouts['default']  = tmp

tmp = {}
sf = 0.9
tmp['figsize']          = (sf*70,sf*30)
tmp['env_rspan']        = 2
tmp['band_rspan']       = 3
tmp['c0_cspan']         = 23
tmp['c1_pos']           = 30
tmp['c1_cspan']         = 75
tmp['nx']               = 100
tmp['map_cbar_shrink']  = 0.75
tmp['freq_size']        = 50
tmp['goes_lw']          = 5
tmp['kp_markersize']    = 20
tmp['title_size']       = 36
tmp['ticklabel_size']   = 28
tmp['label_size']       = 36
tmp['legend_size']      = 24
layouts['2band']        = tmp

tmp = {}
sf = 1.0
tmp['figsize']          = (sf*40,sf*30)
tmp['c0_cspan']         = 23
tmp['c1_pos']           = 27
tmp['c1_cspan']         = 80
tmp['nx']               = 100
tmp['title_size']       = 36
tmp['ticklabel_size']   = 24
tmp['label_size']       = 36
tmp['legend_size']      = 24
tmp['freq_size']        = 60
tmp['freq_xpos']        = -0.120
tmp['goes_lw']          = 5
tmp['kp_markersize']    = 20
tmp['hist_cbar_pad']    = 0.085
tmp['sza_lw']           = 3
tmp['sza_label_size']   = 28
tmp['sza_sun_size']     = 1500
layouts['4band12hr']    = tmp

tmp = {}
sf  = 1.0
szf = 0.75
tmp['figsize']          = (sf*40,sf*30)
tmp['c0_cspan']         = 23
tmp['c1_pos']           = 27
tmp['c1_cspan']         = 80
tmp['nx']               = 100
tmp['title_size']       = 36 * szf
tmp['ticklabel_size']   = 24 * szf
tmp['label_size']       = 24 * szf
tmp['legend_size']      = 24 * szf
tmp['freq_size']        = 60 * szf
tmp['freq_xpos']        = -0.120
tmp['goes_lw']          = 5 * szf
tmp['kp_markersize']    = 20 * szf
tmp['hist_cbar_pad']    = 0.085
tmp['sza_lw']           = 3
tmp['sza_label_size']   = 28 * szf
tmp['sza_sun_size']     = 1500 * szf
layouts['4band24hr']    = tmp

tmp = {}
sf = 1.0
tmp['figsize']          = (sf*40,sf*35)
tmp['c0_cspan']         = 23
tmp['c1_pos']           = 28
tmp['c1_cspan']         = 80
tmp['nx']               = 100
tmp['title_size']       = 36
tmp['ticklabel_size']   = 24
tmp['label_size']       = 24
tmp['legend_size']      = 24
tmp['freq_size']        = 40
tmp['freq_xpos']        = -0.120
tmp['goes_lw']          = 5
tmp['kp_markersize']    = 12
tmp['hist_cbar_pad']    = 0.050
tmp['sza_lw']           = 3
tmp['sza_label_size']   = 26
tmp['sza_sun_size']     = 1500
layouts['6band3day']    = tmp

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
        calc_hist_maxes=False,xlabels=True,plot_title=False,cbar_pad=0.05,
        xb_size_min=10.,yb_size_km=250.,stat=None,plot_type='contour'):
    # TODO: Make all of this stuff configurable
    # Ultimately the goal is for this to be very versatile
    # x-axis: UTC

#    xb_size_min = 30.
#    yb_size_km  = 500.

    ret_dct = {}

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

    if stat is None:
        vm_scale    = 0.6
        cbar_label  = 'Ham Radio\nSpot Density'
        if log_hist:
            tf          = hist >= 1
            tmp         = np.log10(hist[tf])

            hist_1      = hist*0.
            hist_1[tf]  = tmp
            hist        = hist_1
#            vm_scale    = 0.8
            vm_scale    = 1.00
            cbar_label  = 'log10(Ham Radio\nSpot Density)'

        if vmin is None: vmin    = 0
        if vmax is None:
            vmax    = vm_scale*np.max(hist)
            if np.sum(hist) == 0: vmax = 1.0
    else:
        stat_param  = 'snr'

        hist[:] = np.nan
        for xb_inx,xb_0 in enumerate(xb[:-1]):
            xb_1    = xb[xb_inx+1]
            for yb_inx,yb_0 in enumerate(yb[:-1]):
                yb_1    = yb[yb_inx+1]

                tf_x    = np.logical_and(total_hours >= xb_0,
                                         total_hours <  xb_1)

                tf_y    = np.logical_and(df['dist_Km'] >= yb_0,
                                         df['dist_Km'] <  yb_1)

                tf      = np.logical_and(tf_x,tf_y)
                if np.sum(tf) == 0: continue

                val     = df[tf][stat_param].apply(stat)

                hist[xb_inx,yb_inx] = val

        cbar_label  = '{!s}({!s})'.format(stat,stat_param)
        lval_label  = cbar_label 

        vm_scale    = 0.8
        if vmin is None: 
            vmin    = np.nanmin(hist) + vm_scale*np.abs(np.nanmin(hist))
        if vmax is None:
            vmax    = vm_scale*np.nanmax(hist)
            if np.sum(hist) == 0: vmax = 1.0

#        tf          = np.isnan(hist)
#        hist[tf]    = vmin

    ret_dct['hist'] = hist
    if calc_hist_maxes:
        return ret_dct

    if xlabels:
        xdct    = gl.prmd[xkey]
        xlabel  = xdct.get('label',xkey)
        ax.set_xlabel(xlabel)

    if plot_title:
        ax.set_title(title)

    ax.set_ylabel('R_gc [km]')

    levels  = np.linspace(vmin,vmax,15)

    cmap    = plt.cm.viridis
    xb_dt   = [xlim[0]+datetime.timedelta(hours=(x-xbin_0)) for x in xb]
    if plot_type == 'contour':
        norm    = matplotlib.colors.Normalize(vmin=vmin,vmax=vmax)
        pcoll   = ax.contourf(xb_dt[:-1],yb[:-1],hist.T,levels,norm=norm,cmap=cmap)

        cbar    = plt.colorbar(pcoll,ax=ax,pad=cbar_pad)
        cbar.set_label(cbar_label)
    elif plot_type == '2d_hist':
        verts   = []
        vals    = []
        for xb_inx in range(len(xb_dt)-1):
            xb_0    = matplotlib.dates.date2num(xb_dt[xb_inx])
            xb_1    = matplotlib.dates.date2num(xb_dt[xb_inx+1])

            for yb_inx,yb_0 in enumerate(yb[:-1]):
                yb_1    = yb[yb_inx+1]
                x1,y1 = (xb_0,yb_0)
                x2,y2 = (xb_1,yb_0)
                x3,y3 = (xb_1,yb_1)
                x4,y4 = (xb_0,yb_1)
                verts.append(((x1,y1),(x2,y2),(x3,y3),(x4,y4),(x1,y1)))

                vals.append(hist[xb_inx,yb_inx])

        bounds  = np.linspace(vmin,vmax,256)
        norm    = matplotlib.colors.BoundaryNorm(bounds,cmap.N)

        pcoll   = PolyCollection(np.array(verts),edgecolors='face',closed=False,cmap=cmap,norm=norm)
        pcoll.set_array(np.array(vals))
        ax.add_collection(pcoll,autolim=False)

        cbar    = plt.colorbar(pcoll,ax=ax,pad=cbar_pad)
        cbar.set_label(cbar_label)
    elif plot_type == 'line':
        yy  = np.sum(hist,axis=1)
        ax.plot(xb_dt[:-1],yy)

        df_line = pd.DataFrame({'spots':yy},index=xb_dt[:-1])
        ret_dct['df_line'] = df_line
        ylim = None

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    return ret_dct

def plot_on_map(ax,frame,param='mids',cparam=None,box=None,lout=None):
    if param == 'mids':
        xx_param    = 'md_long'
        yy_param    = 'md_lat'

        mcolor      = 'blue'
        marker      = 'o'
        msize       = 5
    elif param == 'tx':
        xx_param    = 'tx_long'
        yy_param    = 'tx_lat'
        frame       = frame.drop_duplicates([xx_param,yy_param]).copy()

        mcolor      = 'black'
        marker      = '.'
        msize       = 1

    elif param == 'rx':
        xx_param    = 'rx_long'
        yy_param    = 'rx_lat'
        frame       = frame.drop_duplicates([xx_param,yy_param]).copy()

        mcolor      = 'blue'
        marker      = 'o'
        msize       = 20
    elif param == 'box_in' or param == 'box_out':
        rgn         = gl.regions.get(box)
        lat_lim     = rgn.get('lat_lim')
        lon_lim     = rgn.get('lon_lim')

        df  = pd.DataFrame()
        for trx in ['rx','tx']:
            dft = frame.copy()
            dft['endpoint_lat']  = dft['{!s}_lat'.format(trx)]
            dft['endpoint_long'] = dft['{!s}_long'.format(trx)]
            df  = df.append(dft,ignore_index=True)

        lats    = df['endpoint_lat']
        lons    = df['endpoint_long']

        tf_lat  = np.logical_and(lats >= lat_lim[0],
                                 lats <  lat_lim[1])

        tf_lon  = np.logical_and(lons >= lon_lim[0],
                                 lons <  lon_lim[1])

        tf  = np.logical_and(tf_lat,tf_lon)

        if param == 'box_out':
            tf = np.logical_not(tf)
        df          = df[tf].copy()

        xx_param    = 'endpoint_long'
        yy_param    = 'endpoint_lat'
        frame       = df.drop_duplicates([xx_param,yy_param]).copy()

        if param == 'box_in':
            mcolor      = 'black'
            marker      = 'o'
            msize       = 50
        else:
            cparam      = 'dist_Km'
            marker      = 'o'
            msize       = 50
    
    xx      = frame[xx_param]
    yy      = frame[yy_param]

    if len(xx) == 0:
        xx  = np.array([0,0])
        yy  = np.array([0,0])

    if cparam is None:
        sct_dct = {}
        sct_dct['color']    = mcolor
        sct_dct['marker']   = marker
        sct_dct['s']        = msize
        sct_dct['zorder']   = 10
        pcoll   = ax.scatter(xx,yy,**sct_dct)
    else:
        cc      = frame[cparam]
        if len(xx) == 0:
            cc  = np.array([0,0])

        cdct    = gl.prmd[cparam]
        cmap    = cdct.get('cmap',matplotlib.cm.inferno)
        vmin    = cdct.get('vmin',0)
        vmax    = cdct.get('vmax',np.nanmax(cc))

        sct_dct = {}
        sct_dct['c']        = cc
        sct_dct['cmap']     = cmap
        sct_dct['vmin']     = vmin
        sct_dct['vmax']     = vmax
        sct_dct['marker']   = marker
        sct_dct['s']        = msize
        sct_dct['zorder']   = 10
        pcoll   = ax.scatter(xx,yy,**sct_dct)

        cb_props = {}
        cb_props['shrink']      = lout.get('map_cbar_shrink',1.)
        cb_props['fraction']    = lout.get('map_cbar_fraction',0.15)
        cb_props['pad']         = lout.get('map_cbar_pad',0.05)
        cb_props['aspect']      = lout.get('map_cbar_aspect',20)
        cbar    = plt.colorbar(pcoll,ax=ax,**cb_props)

        clabel  = cdct.get('label',cparam)
        fontdict = {'size':'xx-large','weight':'normal'}
        cbar.set_label(clabel,fontdict=fontdict)

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

def make_figure(sTime,eTime,xkey='occurred',
        rgc_lim=(0,40000), maplim_region='World', filter_region=None, filter_region_kind='midpoints',
        log_hist=False,output_dir='output',calc_hist_maxes=False,fname=None,box=None,band_obj=None,
        xb_size_min=10.,yb_size_km=250.,stat=None,plot_type='contour',
        map_midpoints=True,map_midpoints_cparam=None,
        map_tx=False,map_tx_cparam=None,
        map_rx=False,map_rx_cparam=None,
        map_filter_region=False,
        solar_zenith_region=None,
        find_flares=False,flare_labels=True,
        plot_summary=False,line_csvName=None,
        layout=None,generate_csv=False,):
    """
    xkey:   {'slt_mid','ut_hrs'}
    """

    mark_times  = []
    if layout is None:
        lout = layouts.get('default')
    else:
        lout = layouts.get(layout)

    set_text_props(**lout)

    if band_obj is None:
        band_obj    = gl.BandData()

    band_dict   = band_obj.band_dict

    print('Loading CSVs...')
    df      = pd.DataFrame()
    dates   = list(daterange(sTime, eTime))
    for dt in tqdm.tqdm(dates):
        dft         = gl.load_spots_csv(dt.strftime("%Y-%m-%d"),rgc_lim=rgc_lim,
                        filter_region=filter_region,filter_region_kind=filter_region_kind)
        df          = df.append(dft,ignore_index=True)

        if generate_csv:
            csv_fname   = '{!s}-RBN_WSPR'.format(dt.strftime("%Y-%m-%d"))
            gen_csv(dft,output_dir,csv_fname)

    if generate_csv:
        return
    tf  = np.logical_and(df['occurred'] >= sTime,
                         df['occurred'] <= eTime+datetime.timedelta(minutes=xb_size_min))
    df  = df[tf].copy()

    date_str_0  = sTime.strftime('%d %b %Y')
    date_str_1  = eTime.strftime('%d %b %Y')
    if eTime-sTime < datetime.timedelta(hours=24):
        date_str    = sTime.strftime('%d %b %Y')
    else:
        date_str    = '{!s} - {!s}'.format(date_str_0,date_str_1)

    sza = None
    # Solar Zenith Calculation #############
    if solar_zenith_region is not None:
        sza,sza_lat,sza_lon = gl.calc_solar_zenith_region(sTime,eTime,region=solar_zenith_region)

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
    if plot_summary:
        ny  += band_rspan
    
    sf      = lout.get('sf',1.00)  # Scale Factor
    figsize = lout.get('figsize',(sf*40, sf*4*len(band_dict)))

    fig = plt.figure(figsize=figsize)

    # Keep track of all of the time series axes.
    ts_axs  = []

    # Geospace Environment ####################
    axs_to_adjust   = []
    omni            = Omni()
    ax              = plt.subplot2grid((ny,nx),(0,c1_pos),colspan=c1_span,rowspan=env_rspan)
    msize           = lout.get('kp_markersize',10)
    dst_lw          = lout.get('goes_lw',2)
    omni_axs        = omni.plot_dst_kp(sTime,eTime,ax,xlabels=True,
                        kp_markersize=msize,dst_lw=dst_lw,dst_param='SYM-H')
    axs_to_adjust   += omni_axs
    for ax in omni_axs:
        ts_axs.append({'ax':ax,'axvline_props':{'color':'k'}})

#    tf  = np.logical_and(omni.df.index >= sTime,
#                         omni.df.index <  eTime)
#
#    odf = omni.df[tf].copy()

    ########################################
    goes_dcts       = OrderedDict()
    goes_dcts[13]   = {}
    goes_dcts[15]   = {}

    flares_combined = pd.DataFrame()
    for sat_nr,gd in goes_dcts.items():
        gd['data']      = goes.read_goes(sTime,eTime,sat_nr=sat_nr)
        flares          = goes.find_flares(gd['data'],min_class='M1',window_minutes=60)
        flares['sat']   = sat_nr
        gd['flares']    = flares
        flares_combined = flares_combined.append(flares).sort_index()
        gd['var_tags']  = ['B_AVG']
        gd['labels']    = ['GOES {!s}'.format(sat_nr)]

    flares_combined     = flares_combined[~flares_combined.index.duplicated()].sort_index()
    csv_path            = os.path.join(output_dir,'{!s}-flares'.format(date_str))  
    with open(csv_path+'.txt','w') as fl:
        fl.write(flares_combined.to_string())
    flares_combined.to_csv(csv_path+'.csv')

    ax              = plt.subplot2grid((ny,nx),(env_rspan,c1_pos),colspan=c1_span,rowspan=env_rspan)
    xdct            = gl.prmd[xkey]
    xlabel          = xdct.get('label',xkey)
    goes_lw         = lout.get('goes_lw',2)
    for sat_nr,gd in goes_dcts.items():
        goes.goes_plot(gd['data'],sTime,eTime,ax=ax,
                var_tags=gd['var_tags'],labels=gd['labels'],
                legendLoc='upper right',lw=goes_lw)

        if find_flares:
            flares  = gd['flares']
#            with open(os.path.join(output_dir,'{!s}-G{!s}-flares.txt'.format(date_str,sat_nr)),'w') as fl:
#                fl.write(flares.to_string())

            for key,flare in flares.iterrows():
                flr_plt_dct = {}
#                flr_plt_dct['label']        = '{0} Class Flare @ {1}'.format(flare['class'],key.strftime('%H%M UT'))
                flr_plt_dct['color']        = gd.get('color','blue')
                flr_plt_dct['marker']       = gd.get('marker','o')
                flr_plt_dct['markersize']   = gd.get('markersize',10)
                ax.plot(key,flare['B_AVG'],**flr_plt_dct)
                label   = '{!s} ({!s})'.format(key.strftime('%H%M UT'),flare['class'])
                mark_times.append({'val':key,'label':label})

    title   = 'NOAA GOES X-Ray (0.1 - 0.8 nm) Irradiance'
    size    = lout.get('label_size',20)
    ax.text(0.01,0.05,title,transform=ax.transAxes,ha='left',fontdict={'size':size,'weight':'bold'})
    axs_to_adjust.append(ax)
    ts_axs.append({'ax':ax,'axvline_props':{'color':'k'}})

    # Summary Plot #########################
    if plot_summary:
        fig_row = n_env*env_rspan
        tfs     = []
        bstrs   = []
        for band_inx, (band_key,band) in enumerate(band_dict.items()):
            tf = df["band"] == band.get('meters')
            tfs.append(tf)

            bstr    = '{!s}'.format(band['freq'])
            if band_inx == len(band_dict)-1:
                bstr = '& ' + bstr

            bstrs.append(bstr)

        if len(bstrs) == 2:
            bstring = ' '.join(bstrs)
        else:
            bstring = ', '.join(bstrs)
        bstring = bstring + ' MHz'

        frame = df[np.logical_or.reduce(tfs)].copy()
        frame.sort_values(xkey,inplace=True)

        n_mids  = len(frame)
        print('   {!s}: Summary: {!s} (n={!s})'.format(date_str,bstring,n_mids))

        # Histograms ########################### 
        ax      = plt.subplot2grid((ny,nx),(fig_row,c1_pos),
                    colspan=c1_span,rowspan=band_rspan)
        axs_to_adjust.append(ax)
        ts_axs.append({'ax':ax,'axvline_props':{'color':'w'}})
        title   = '{!s} ({!s})'.format(date_str,bstring)

        vmin    = None
        vmax    = None

        pad     = lout.get('hist_cbar_pad',0.05)
        hist_dct = make_histogram_from_dataframe(frame, ax, title,xkey=xkey,xlim=(sTime,eTime),ylim=rgc_lim,
                    vmin=vmin,vmax=vmax,calc_hist_maxes=calc_hist_maxes,xlabels=False,log_hist=log_hist,
                    cbar_pad=pad,xb_size_min=xb_size_min,yb_size_km=yb_size_km,stat=stat,plot_type='line')

        hist    = hist_dct.get('hist')
        df_line = hist_dct.get('df_line')
        if df_line is not None:
            if line_csvName is None:
                line_csvName = '{!s}_{!s}_{!s}_map-{!s}_filter-{!s}-{!s}-DFLINE-{!s}.csv'.format(
                        date_str,xkey, rgc_lim, maplim_region, filter_region, filter_region_kind,
                        datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')
                        )
            csv_fpath   = os.path.join(output_dir,line_csvName)
            df_line.to_csv(csv_fpath)

        fdict       = {'size':lout.get('freq_size',35),'weight':'bold'}
        freq_xpos   = lout.get('freq_xpos',-0.075)
        ax.text(freq_xpos,0.5,'All Bands',transform=ax.transAxes,va='center',rotation=90,fontdict=fdict)

        # Solar Zenith Angle
        if sza is not None:
            sza_ax  = ax.twinx()
            sza_ax.plot(sza.index,sza.els,ls='--',lw=lout.get('sza_lw',2),color='white')
            ylabel  = u'Solar Zenith \u2220\n@ ({:.0f}\N{DEGREE SIGN} N, {:.0f}\N{DEGREE SIGN} E)'.format(sza_lat,sza_lon)

            fontdict    = {}
            fsize       = lout.get('sza_label_size')
            if fsize is not None:
                fontdict['size'] = fsize
            sza_ax.set_ylabel(ylabel,fontdict=fontdict)
            sza_ax.set_ylim(110,0)

        #    # Map ################################## 
        ax = plt.subplot2grid((ny,nx),(fig_row,0),projection=ccrs.PlateCarree(),
                rowspan=band_rspan,colspan=c0_span)
        ax.coastlines()
        ax.gridlines()

        if map_midpoints:
            plot_on_map(ax,frame,param='mids',cparam=map_midpoints_cparam,lout=lout)

        if map_tx:
            plot_on_map(ax,frame,param='tx',cparam=map_tx_cparam,lout=lout)

        if map_rx:
            plot_on_map(ax,frame,param='rx',cparam=map_rx_cparam,lout=lout)

        if map_filter_region:
            plot_on_map(ax,frame,param='box_out',box=filter_region,lout=lout)
            plot_on_map(ax,frame,param='box_in',box=filter_region,lout=lout)

        if box is not None:
            rgn = gl.regions.get(box)
            x0  = rgn['lon_lim'][0]
            y0  = rgn['lat_lim'][0]
            ww  = rgn['lon_lim'][1] - x0
            hh  = rgn['lat_lim'][1] - y0
            
            p   = matplotlib.patches.Rectangle((x0,y0),ww,hh,fill=False,zorder=500)
            ax.add_patch(p)

        if sza is not None:
            sun_size    = lout.get('sza_sun_size',600)
            ax.scatter([sza_lon],[sza_lat],marker='*',s=sun_size,color='yellow',
                            edgecolors='black',zorder=500,lw=3)

        ax.set_xlim(gl.regions[maplim_region]['lon_lim'])
        ax.set_ylim(gl.regions[maplim_region]['lat_lim'])
        label   = 'Radio Spots (N = {!s})'.format(n_mids)
        fdict   = {'size':lout.get('label_size',24)}
        ax.text(0.5,-0.15,label,transform=ax.transAxes,ha='center',fontdict=fdict)

    ########################################
    hist_maxes  = {}
    for band_inx, (band_key,band) in enumerate(band_dict.items()):
        fig_row = n_env*env_rspan + band_inx*band_rspan
        if plot_summary:
            fig_row += band_rspan
        if band_inx == len(band_dict)-1:
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
        ts_axs.append({'ax':ax,'axvline_props':{'color':'w'}})
        title   = '{!s} ({!s})'.format(date_str,band.get('freq_name'))

        vmin    = band.get('vmin')
        vmax    = band.get('vmax')

        pad     = lout.get('hist_cbar_pad',0.05)
        hist    = make_histogram_from_dataframe(frame, ax, title,xkey=xkey,xlim=(sTime,eTime),ylim=rgc_lim,
                    vmin=vmin,vmax=vmax,calc_hist_maxes=calc_hist_maxes,xlabels=xlabels,log_hist=log_hist,
                    cbar_pad=pad,xb_size_min=xb_size_min,yb_size_km=yb_size_km,stat=stat,plot_type=plot_type)

        fdict       = {'size':lout.get('freq_size',35),'weight':'bold'}
        freq_xpos   = lout.get('freq_xpos',-0.075)
        ax.text(freq_xpos,0.5,band.get('freq_name'),transform=ax.transAxes,va='center',rotation=90,fontdict=fdict)

        # Solar Zenith Angle
        if sza is not None:
            sza_ax  = ax.twinx()
            sza_ax.plot(sza.index,sza.els,ls='--',lw=lout.get('sza_lw',2),color='white')
            ylabel  = u'Solar Zenith \u2220\n@ ({:.0f}\N{DEGREE SIGN} N, {:.0f}\N{DEGREE SIGN} E)'.format(sza_lat,sza_lon)

            fontdict    = {}
            fsize       = lout.get('sza_label_size')
            if fsize is not None:
                fontdict['size'] = fsize
            sza_ax.set_ylabel(ylabel,fontdict=fontdict)
            sza_ax.set_ylim(110,0)

        hist_ax = ax

        if calc_hist_maxes:
            hist_maxes[band_key]    = np.max(hist)
            continue
        
        #    # Map ################################## 
        ax = plt.subplot2grid((ny,nx),(fig_row,0),projection=ccrs.PlateCarree(),
                rowspan=band_rspan,colspan=c0_span)

        ax.coastlines()
        ax.gridlines()

        if map_midpoints:
            plot_on_map(ax,frame,param='mids',cparam=map_midpoints_cparam,lout=lout)

        if map_tx:
            plot_on_map(ax,frame,param='tx',cparam=map_tx_cparam,lout=lout)

        if map_rx:
            plot_on_map(ax,frame,param='rx',cparam=map_rx_cparam,lout=lout)

        if map_filter_region:
            plot_on_map(ax,frame,param='box_out',box=filter_region,lout=lout)
            plot_on_map(ax,frame,param='box_in',box=filter_region,lout=lout)

        if box is not None:
            rgn = gl.regions.get(box)
            x0  = rgn['lon_lim'][0]
            y0  = rgn['lat_lim'][0]
            ww  = rgn['lon_lim'][1] - x0
            hh  = rgn['lat_lim'][1] - y0
            
            p   = matplotlib.patches.Rectangle((x0,y0),ww,hh,fill=False,zorder=500)
            ax.add_patch(p)

        if sza is not None:
            sun_size    = lout.get('sza_sun_size',600)
            ax.scatter([sza_lon],[sza_lat],marker='*',s=sun_size,color='yellow',
                            edgecolors='black',zorder=500,lw=3)

        ax.set_xlim(gl.regions[maplim_region]['lon_lim'])
        ax.set_ylim(gl.regions[maplim_region]['lat_lim'])
        label   = 'Radio Spots (N = {!s})'.format(n_mids)
        fdict   = {'size':lout.get('label_size',24)}
        ax.text(0.5,-0.15,label,transform=ax.transAxes,ha='center',fontdict=fdict)

    if calc_hist_maxes:
        plt.close(fig)
        return hist_maxes

    # Mark flare times with a vertical line on each axis.
    for axd in ts_axs:
        ax      = axd.get('ax')
        prp     = axd.get('axvline_props',{})
        color   = prp.get('color','k')
        lw      = prp.get('lw',2)
        ls      = prp.get('ls','--')
        for mtd in mark_times:
            mark_time   = mtd.get('val')
            label       = mtd.get('label')

            ax.axvline(mark_time,color=color,lw=lw,ls=ls)
            if flare_labels:
                trans   = matplotlib.transforms.blended_transform_factory(ax.transData, ax.transAxes)
                fsize   = lout.get('flarelabel_size',22)
                ax.text(mark_time,1,label,rotation=90,va='top',ha='right',transform=trans,fontdict={'weight':'bold','size':fsize},color=color)

    fig.subplots_adjust(left=0,right=1,bottom=0,top=1)

    # Force geospace environment axes to line up with histogram
    # axes even though it doesn't have a color bar.
    for ax_0 in axs_to_adjust:
        gl.adjust_axes(ax_0,hist_ax)

#    fdict   = {'size':50,'weight':'bold'}
#    title   = '{!s}-\n{!s}'.format(date_str_0,date_str_1)
#    fig.text(0.030,0.925,title,fontdict=fdict)

    xpos    = lout.get('main_title_xpos',0.150) # 0.030
    ypos    = lout.get('main_title_ypos',0.800) # 0.925
    fdict   = {'size':50,'weight':'bold'}
    if eTime-sTime < datetime.timedelta(hours=24):
        title   = date_str_0
    else:
        title   = '{!s}-\n{!s}'.format(date_str_0,date_str_1)
    fig.text(xpos,ypos,title,fontdict=fdict)

    meters  = [x['meters'] for x in band_obj.band_dict.values()]
    srcs    = '\n'.join([' '+x for x in gl.list_sources(df,bands=meters)])
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
    gl.prep_output({0:output_dir},clear=False)
    test_configuration  = True
    global_cbars        = False

    run_dcts    = []
    dct = {}
    dct['sTime']                = datetime.datetime(2017, 9, 4)
    dct['eTime']                = datetime.datetime(2017, 9, 14)
    dct['rgc_lim']              = (0,5000)
    dct['maplim_region']        = 'Greater Greater Caribbean'
    dct['box']                  = 'Greater Caribbean'
    dct['solar_zenith_region']  = dct['box']
    dct['filter_region']        = dct['box']
    dct['filter_region_kind']   = 'endpoints'
    dct['log_hist']             = True
    dct['band_obj']             = gl.BandData([7,14])
    dct['map_midpoints']        = False
    dct['map_filter_region']    = True
    dct['layout']               = '2band'
    dct['output_dir']           = output_dir
    dct['fname']                = 'caribbean'
    dct['find_flares']          = False
    dct['flare_labels']         = False
    run_dcts.append(dct)


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
