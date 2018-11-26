#!/usr/bin/env python3
import os
import glob
import datetime
from collections import OrderedDict
import multiprocessing as mp

import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
from matplotlib import pyplot as plt

import mysql.connector

import tqdm
import seqp
from seqp import calcSun
import eclipse_calc

import cartopy.io.shapereader as shapereader
import shapely.geometry as sgeom

from shapely.ops import unary_union
from shapely.prepared import prep

land_shp_fname = shapereader.natural_earth(resolution='110m',
                                       category='physical', name='land')

land_geom   = unary_union(list(shapereader.Reader(land_shp_fname).geometries()))
land        = prep(land_geom)

def generate_sim_jobs(df,jobs_dir='eclipse_sim/jobs',clear_dir=True):
    #2017-08-21 16:51:00.csv
    #RX_CALL,TX_CALL,MHZ,RX_LAT,RX_LON,TX_LAT,TX_LON
    #KU7T,N2BA-2,7.03,47.48,-121.79,43.69,-116.29
    if clear_dir:
        jobs_fls    = glob.glob(os.path.join(jobs_dir,'*.csv'))
        for jobs_fl in jobs_fls:
            os.remove(jobs_fl)


    # Generate list of valid model times.
    sDate   = datetime.datetime(2017,8,21,16)
    eDate   = datetime.datetime(2017,8,21,22)
    dt      = datetime.timedelta(minutes=3)
    model_times = [sDate]
    while model_times[-1] < eDate:
        model_times.append(model_times[-1] + dt)
    model_times = np.array(model_times)

    keys = OrderedDict()
    keys['call_0']      = 'RX_CALL'
    keys['call_1']      = 'TX_CALL'
    keys['frequency']   = 'MHZ'
    keys['lat_0']       = 'RX_LAT'
    keys['lon_0']       = 'RX_LON'
    keys['lat_1']       = 'TX_LAT'
    keys['lon_1']       = 'TX_LON'

    jobs    = []
    dts     = set([x.to_pydatetime() for x in df.obs_max_dt_mid])
    for time in dts:
        tf      = df.obs_max_dt_mid == time
        df_0    = df[tf].copy()
        
        df_0    = df_0[list(keys.keys())]
        df_0    = df_0.rename(columns=keys)

        inx         = np.abs(model_times-time).argmin()
        model_time  = model_times[inx]

        fname   = model_time.strftime('%Y-%m-%d %H:%M:%S.csv')
        fpath   = os.path.join(jobs_dir,fname)
        df_0.to_csv(fpath,index=False)
        jobs.append(fpath)
    return jobs

def calculate_Razm(df):
    R_gc            = seqp.geopack.greatCircleDist(df.lat_0,df.lon_0,df.lat_1,df.lon_1) * Re
    azm             = seqp.geopack.greatCircleAzm(df.lat_0,df.lon_0,df.lat_1,df.lon_1)

    df['R_gc']      = R_gc
    df['azm']       = azm
    return df

def calculate_midpoint(df):
    mids            = seqp.geopack.midpoint(df.lat_0,df.lon_0,df.lat_1,df.lon_1)
    df['lat_mid']   = mids[0]
    df['lon_mid']   = mids[1]
    return df

def is_land(lonlat):
    return land.contains(sgeom.Point(lonlat[0], lonlat[1]))

#MAP_RES  = '110m'
#MAP_TYPE = 'physical'
#MAP_NAME = 'land'
#
#shape_data = shapereader.natural_earth(resolution=MAP_RES, category=MAP_TYPE, name=MAP_NAME)
#lands = shapereader.Reader(shape_data).geometries()

## Check if a point is over land.
#def is_over_land(lat, lon):
#    for land in lands:
#        if land.contains(sgeom.Point(lat, lon)): return True
#
#    # If it wasn't found, return False.
#    return False


def tx_on_land(df):
    # Make sure all virtual transmitters are on land.
    lats    = df['lat_1'].tolist()
    lons    = df['lon_1'].tolist()

    print('Computing on land...')
    lonlats = [*zip(lons,lats)]

    unq     = [*set(lonlats)]
    tf      = [*map(is_land,unq)]
    land    = [i for (i,v) in zip(unq,tf) if v]

    tf      = [*map(lambda x: x in land, lonlats)]
    
    df      = df[tf]
    return df


prep_output = seqp.gen_lib.prep_output
php         = False

rcp = mpl.rcParams
rcp['figure.titlesize']     = 'xx-large'
rcp['axes.titlesize']       = 'xx-large'
rcp['axes.labelsize']       = 'xx-large'
rcp['xtick.labelsize']      = 'xx-large'
rcp['ytick.labelsize']      = 'xx-large'
rcp['legend.fontsize']      = 'large'

rcp['figure.titleweight']   = 'bold'
rcp['axes.titleweight']     = 'bold'
rcp['axes.labelweight']     = 'bold'

Re  = 6371.
hgt = 300.

base_dir    = os.path.join('output',os.path.basename(__file__)[:-3])
prep_output({0:base_dir},clear=False,php=php)

cache_dir    = os.path.join('cache',os.path.basename(__file__)[:-3])
prep_output({0:cache_dir},clear=False,php=False)

# Define Bands
bandObj     = seqp.maps.BandData()
bands = [1,3,7,14,21,28]

# Sources

sources   = OrderedDict()
dsd = {}
data_set = 'SEQP RBN Observations'
sources[data_set]       = dsd
dsd['csv_in']           = 'data/seqp_all_spots/north_america_rbn_filtered.csv.bz2'
dsd['cache_csv']        = os.path.join(cache_dir,'{!s}.csv.bz2'.format(data_set))

dsd = {}
data_set = 'control'
sources[data_set]       = dsd
dsd['csv_in']           = 'data/control/rbn_control_northAmerica_obscuration.csv.bz2'
dsd['cache_csv']        = os.path.join(cache_dir,'{!s}.csv.bz2'.format(data_set))
#dsd['date']             = None

dsd = {}
data_set = 'Eclipse Simulation'
sources[data_set]       = dsd
dsd['csv_in']           = 'data/eclipsesim_output_grl_2018/eclipse_traces/eclipse_sim_all_spots.csv.bz2'
dsd['cache_csv']        = os.path.join(cache_dir,'{!s}.csv.bz2'.format(data_set))

dsd = {}
data_set = 'NovemberSS_2017'
sources[data_set]       = dsd
dsd['csv_in']           = 'data/control/rbn_control_201705_1400_2200_NovSS2017.csv.bz2'
dsd['cache_csv']        = os.path.join(cache_dir,'{!s}.csv.bz2'.format(data_set))

# Parameter Dictionary
prmd = {}

tmp = {}
tmp['label']            = 'Midpoint Obscuration at 300 km Altitude'
tmp['lim']              = (0,1.05)
prmd['obs_mid_300km']   = tmp

tmp = {}
tmp['label']            = 'Mean Great Circle Hop Length [km]'
tmp['lim']              = (0.,4500.)
prmd['R_gc_mean']       = tmp

tmp = {}
tmp['label']            = 'Great Circle Distance [km]'
#tmp['lim']              = (0.,17500.)
tmp['lim']              = (0.,8000.)
#tmp['lim']              = (0.,5000.)
tmp['vmin']             = 0.
tmp['vmax']             = 10000.
prmd['R_gc']       = tmp

tmp = {}
tmp['label']            = 'Frequency [MHz]'
tmp['vmin']             = 0.
tmp['vmax']             = 30.
tmp['cmap']             = mpl.cm.jet
prmd['frequency']       = tmp

tmp = {}
tmp['label']            = 'Solar Local Time [hr]'
tmp['vmin']             = 6.
tmp['vmax']             = 18.
tmp['cmap']             = mpl.cm.jet
prmd['slt_mid']         = tmp

tmp = {}
tmp['label']            = 'SNR [dB]'
tmp['lim']              = (0.,100.)
tmp['vmin']             = 20.
tmp['vmax']             = 50.
tmp['cmap']             = mpl.cm.viridis
prmd['srpt_0']          = tmp

tmp = {}
tmp['label']            = 'N Hops'
tmp['vmin']             = 0.
tmp['vmax']             = 5.
tmp['cmap']             = mpl.cm.jet
prmd['N_hops']          = tmp

tmp = {}
tmp['key']              = 'nvals'
tmp['label']            = 'N'
#tmp['lim']              = (0.,750.)
prmd['nvals']           = tmp

bin_scale   = 1.
#bin_scale   = 0.5

tmp = {}
tmp['key']              = 'srpt_0'
tmp['label']            = 'dB'
#tmp['lim']              = (0.,50.)
tmp['lim']              = (0.,200.)
tmp['plot_scatter']     = True
tmp['bin_size']         = 10.*bin_scale
tmp['bins']             = np.arange(tmp['lim'][0],
                                    tmp['lim'][1]+2*tmp['bin_size'],
                                    tmp['bin_size'])
prmd['max_snr']         = tmp

tmp = {}
tmp['key']              = 'R_gc'
tmp['label']            = 'R_gc [km]'
tmp['lim']              = (0.,3000.)
tmp['plot_scatter']     = True
tmp['bin_size']         = 500.*bin_scale
tmp['bins']             = np.arange(tmp['lim'][0],
                                    tmp['lim'][1]+2*tmp['bin_size'],
                                    tmp['bin_size'])
prmd['max_rgc']         = tmp

tmp = {}
tmp['key']              = 'ut_hrs'
tmp['label']            = 'UT Hours'
tmp['lim']              = (14, 22)
tmp['bin_size']         = 10./60. * bin_scale
tmp['bins']             = np.arange(tmp['lim'][0],
                                    tmp['lim'][1]+2*tmp['bin_size'],
                                    tmp['bin_size'])
prmd['ut_hrs']          = tmp

tmp = {}
tmp['key']              = 'epoch'
tmp['label']            = 'Epoch Hours'
tmp['lim']              = (-1.5,1.5)
#tmp['lim']              = (-2,2)
#tmp['lim']              = (-4,4)
tmp['bin_size']         = 10./60. * bin_scale
tmp['bins']             = np.arange(tmp['lim'][0],
                                    tmp['lim'][1]+2*tmp['bin_size'],
                                    tmp['bin_size'])
prmd['epoch']           = tmp

class MySqlEclipse(object):
    def __init__(self,user='hamsci',password='hamsci',host='localhost',database='seqp_analysis'):
        db          = mysql.connector.connect(user=user, password=password,host=host, database=database,buffered=True)
        crsr        = db.cursor()

        qry         = '''
                      CREATE TABLE IF NOT EXISTS eclipse_times (
                      lat DECIMAL(10,4),
                      lon DECIMAL(10,4),
                      height INT,
                      partial_start DATETIME,
                      partial_start_obs FLOAT,
                      ecl_maxtime DATETIME,
                      ecl_maxtime_obs FLOAT,
                      partial_end DATETIME,
                      partial_end_obs FLOAT
                      );
                      '''
        crsr.execute(qry)
        db.commit()

        self.db     = db
mysql_ecl = MySqlEclipse()

def lim_tf(vals,lim):
    """
    Return True/False vectors for limits that may include
    None or Nan.
    """
    tf  = np.ones(len(vals),dtype=np.bool)

    if not pd.isnull(lim[0]):
        tf_0    = vals >= lim[0]
        tf      = np.logical_and(tf,tf_0)

    if not pd.isnull(lim[1]):
        tf_1    = vals < lim[1]
        tf      = np.logical_and(tf,tf_1)

    return tf

def lim_formatter(lim):
    """
    Return True/False vectors for limits that may include
    None or Nan.
    """

    if pd.isnull(lim[0]) and pd.isnull(lim[1]):
        txt = 'All Values'
    elif pd.isnull(lim[0]):
        txt = '< {!s}'.format(lim[1])
    elif pd.isnull(lim[1]):
        txt = u'\u2265 {!s}'.format(lim[0])
    else:
        txt = '({!s}, {!s})'.format(*lim)

    return txt

def get_eclipse_times_dict():
    user        = 'hamsci'
    password    = 'hamsci'
    host        = 'localhost'
    database    = 'seqp_analysis'
    db          = mysql.connector.connect(user=user,password=password,host=host,database=database,buffered=True)
    
#    slat    = '{:.4F}'.format(lat)
#    slon    = '{:.4F}'.format(lon)
#    sheight = '{:.0F}'.format(height/1000.)

    qry     = ("SELECT * FROM eclipse_times ")
               
    crsr    = db.cursor()
    crsr.execute(qry)
    results = crsr.fetchall()
    crsr.close()

    eclipse_dict = {}
    for result in results:
        #     (          lat,            lon,            hgt )
        key = (str(result[0]), str(result[1]), str(result[2]))

        ecl_times   = {}
        ecl_times['partial_start']      = result[3]
        ecl_times['partial_start_obs']  = result[4]
        ecl_times['ecl_maxtime']        = result[5]
        ecl_times['ecl_maxtime_obs']    = result[6]
        ecl_times['partial_end']        = result[7]
        ecl_times['partial_end_obs']    = result[8]

        eclipse_dict[key] = ecl_times

    return eclipse_dict
eclipse_dict    = get_eclipse_times_dict()

def get_eclipse_times(lat,lon,height=300e3,
            date_0=datetime.datetime(2017,8,21,14),
            date_1=datetime.datetime(2017,8,21,22),
            dt=datetime.timedelta(minutes=2),verbose=False):

    slat    = '{:.4F}'.format(lat)
    slon    = '{:.4F}'.format(lon)
    sheight = '{:.0F}'.format(height/1000.)

    key     = (slat,slon,sheight)
    result  = eclipse_dict.get(key)
    if result is not None:
        return result

    user        = 'hamsci'
    password    = 'hamsci'
    host        = 'localhost'
    database    = 'seqp_analysis'
    db          = mysql.connector.connect(user=user,password=password,host=host,database=database,buffered=True)
    
    qry     = ("SELECT partial_start,partial_start_obs,ecl_maxtime,ecl_maxtime_obs,partial_end,partial_end_obs FROM eclipse_times "
               "WHERE lat={} AND lon={} and height={}".format(slat,slon,sheight))
    crsr    = db.cursor()
    crsr.execute(qry)
    result  = crsr.fetchone()
    crsr.close()

    if result is None:
        if verbose:
            print('Eclipse time record not found... calculating. ({!s}, {!s}, {!s})'.format(lat,lon,height))
        times       = []
        this_time   = date_0
        while this_time < date_1:
            times.append(this_time)
            this_time += dt

        obs     = eclipse_calc.calculate_obscuration(times,lat,lon,height)
        times   = np.array(times)

        # Find where there is some eclipse happening.
        tf      = obs > 0
        obs     = obs[tf]
        times   = times[tf]

        # Find time of partial start, max eclipse, and partial end.
        argmax              = np.nanargmax(obs)
        ecl_maxtime              = times[argmax]
        ecl_maxtime_obs          = float(obs[argmax])

        partial_start       = times[0]
        partial_start_obs   = float(obs[0])
        partial_end         = times[-1]
        partial_end_obs     = float(obs[-1])

        # Add information to mysql database
        add_eclTimes    = ("INSERT INTO eclipse_times "
                           "(lat,lon,height,partial_start,partial_start_obs,ecl_maxtime,ecl_maxtime_obs,partial_end,partial_end_obs)"
                           "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)")

        p_start_str     = partial_start.strftime('%Y-%m-%d %H:%M:%S')
        ecl_maxtime_str = ecl_maxtime.strftime('%Y-%m-%d %H:%M:%S')
        p_end_str       = partial_end.strftime('%Y-%m-%d %H:%M:%S')

        data_eclTimes   = (slat,slon,sheight,
                           p_start_str,     partial_start_obs,
                           ecl_maxtime_str, ecl_maxtime_obs,
                           p_end_str,       partial_start_obs)

        crsr    = db.cursor()
        crsr.execute(add_eclTimes,data_eclTimes)
        db.commit()
        crsr.close()

        ecl_times   = {}
        ecl_times['partial_start']      = partial_start
        ecl_times['partial_start_obs']  = partial_start_obs
        ecl_times['ecl_maxtime']        = ecl_maxtime
        ecl_times['ecl_maxtime_obs']    = ecl_maxtime_obs
        ecl_times['partial_end']        = partial_end
        ecl_times['partial_end_obs']    = partial_end_obs

        eclipse_dict[key]   = ecl_times
    else:
        if verbose:
            print('Using MySQL cached EclipseTime record. ({!s}, {!s}, {!s})'.format(lat,lon,height))
        ecl_times   = {}
        ecl_times['partial_start']      = result[0]
        ecl_times['partial_start_obs']  = result[1]
        ecl_times['ecl_maxtime']        = result[2]
        ecl_times['ecl_maxtime_obs']    = result[3]
        ecl_times['partial_end']        = result[4]
        ecl_times['partial_end_obs']    = result[5]

    db.close()
    return ecl_times

def get_eclipse_times_tuple(args):
    return get_eclipse_times(*args)

def calc_obsc_curve(lat,lon,height=300e3):
    xprm            = prmd['ut_hrs']
    hr_0,hr_1       = xprm.get('lim')
    dt_hr           = xprm.get('bin_size')
    ut_hrs          = xprm.get('bins')

    times   = []
    for ut_hr in ut_hrs:
        hr  = int(ut_hr)
        mnd = (ut_hr-hr)*60.
        mn  = int(mnd)
        sc  = int((mnd-mn)*60.)

        time    = datetime.datetime(2017,8,21,hr,mn,sc)
        times.append(time)

    obsc    = eclipse_calc.calculate_obscuration(times,lat,lon)
    df_obsc = pd.DataFrame({'datetime':times,'obsc':obsc},index=ut_hrs)
    return df_obsc

class CalcObscuration(object):
    def __init__(self):
        self.obsc_dict  = {}
        pass

    def __call__(self,call_0,lat,lon,height=300e3):
        obsc    = self.obsc_dict.get(call_0)
        if obsc is None:
            obsc_dir    = os.path.join('cache','obscuration')
            cfl         = os.path.join(obsc_dir,'{!s}-obscuration.csv'.format( clean_call(call_0) ))
            prep_output({0:obsc_dir},clear=False,php=php)
            # Calculate Obscuration Curves #########  
            if not os.path.exists(cfl):
                print('Compute Obscuration: {!s}'.format(call_0))
                obsc    = calc_obsc_curve(lat,lon)
                obsc.to_csv(cfl) 
            else:
                print('Cached Obscuration: {!s} ({!s})'.format(call_0,cfl))
                obsc = pd.read_csv(cfl,parse_dates=[1],index_col=0)
            self.obsc_dict[call_0] = obsc
        return obsc
calc_obscuration = CalcObscuration()

class TimeCheck(object):
    def __init__(self,comment='TimeCheck Init'):
        txt = '{!s} - {!s}'.format(datetime.datetime.now(),comment)
        print(txt)

    def start(self,comment):
        self.t0 = datetime.datetime.now()
        self.c0 = comment
        print('TimeCheck Start: - {!s}'.format(comment))

    def end(self):
        comment = self.c0
        t0  = self.t0
        t1  = datetime.datetime.now()
        td  = t1 - t0
        print('TimeCheck End: - {!s} ({:0.3f})'.format(comment,td.total_seconds()))

def sunAzEl(dates,lat,lon):
    azs, els = [], []
    for date in dates:
        jd    = calcSun.getJD(date) 
        t     = calcSun.calcTimeJulianCent(jd)
        ut    = ( jd - (int(jd - 0.5) + 0.5) )*1440.
        az,el = calcSun.calcAzEl(t, ut, lat, lon, 0.)
        azs.append(az)
        els.append(el)
    return azs,els

def clean_call(call):
    return call.replace('/','-')

def dt2decimalhours(dt):
    try:
        hrs = dt.hour + dt.minute/60. + dt.second/3600.
    except:
        hrs = np.nan
    return hrs

def plot_subplot(param,df_dct,band,ax,lw=4,smoothing=0,
                    obsc_df=None,solar_df=None,fit=None,xparam='ut_hrs',
                    obsc_lw=6,legend=True,cbar_pad=0.140,sza_pad=1.10):
    xprm            = prmd[xparam]
    xkey            = xprm.get('key',xparam)
    xbin_size       = xprm.get('bin_size')
    xbins           = xprm.get('bins')

#    xbins   = xbins.tolist() + [xbins[-1]+xbin_size]

    yprm            = prmd[param]
    ylim            = yprm.get('lim')
    ylabel          = yprm.get('label')
    key             = yprm.get('key',param)
    plot_scatter    = yprm.get('plot_scatter',False)
    ybin_size       = yprm.get('bin_size')
    ybins           = yprm.get('bins')

    # Select data and dataset
    # (left over from overplotting eclipse/control data)
    data_set,dct    = next(iter(df_dct.items()))

    hrs  = dct[band].get(xkey)
    
    if key == 'nvals':
        vals    = dct[band][key]
        ax.plot(hrs,vals,label=data_set,lw=lw)

    if smoothing:
        vals    = dct[band][key]
        tmp_df  = pd.DataFrame({'vals':vals},index=hrs)
        tmp_df  = tmp_df.rolling(smoothing).mean()
        ax.plot(tmp_df.index,tmp_df.vals,lw=lw)

    fontdict = {'size':'xx-large','weight':'normal'}
    labelpad = 7.5

#    plot_scatter = False
    if plot_scatter:
        xx  = np.array(dct[band]['df'][xkey].tolist())
        yy  = np.array(dct[band]['df'][key].tolist())

        hist,xb,yb  = np.histogram2d(xx,yy,bins=(xbins,ybins))
        
        vmin    = 0
        vmax    = 0.8*np.max(hist)
        if np.sum(hist) == 0:
            vmax = 1.0

        cmap    = plt.cm.jet
        pcoll   = ax.contourf(xb[:-1],yb[:-1],hist.T,15,vmin=vmin,vmax=vmax,cmap=cmap)
#        CS      = ax.contour(xb[:-1],yb[:-1],hist.T,15, linewidths=0.5, colors='k')
#        pcoll   = ax.pcolormesh(xb,yb,hist.T,vmin=vmin,vmax=vmax,cmap=cmap)
#        ax.scatter(xx,yy,color='0.8',alpha=0.5)

        cbar    = plt.colorbar(pcoll,ax=ax,pad=cbar_pad)
        cbar.set_label('Spot Density',fontdict=fontdict,labelpad=labelpad)

        props   = dict(facecolor='white', alpha=0.75)
        txt     = 'N = {!s}'.format(len(yy))
        ax.text(0.015,0.05,txt,transform=ax.transAxes,bbox=props,zorder=500)

    plot_elevation = False
#    plot_elevation = True
    if plot_elevation:
        df  = dct[band]['df']
        if 'initial_elev' in df.keys():
            xx  = np.array(df[xkey].tolist())
            yy  = np.array(df[key].tolist())

            hist,xb,yb  = np.histogram2d(xx,yy,bins=(xbins,ybins))

            zz  = hist*np.nan
            for xinx,xbin in enumerate(xb[:-1]):
                for yinx,ybin in enumerate(yb[:-1]):
                    xtf             = np.logical_and(xx >= xbin, xx < xb[xinx+1])
                    ytf             = np.logical_and(yy >= ybin, yy < yb[yinx+1])
                    tf              = np.logical_and(xtf,ytf)
                    val             = df['initial_elev'][tf].mean()
                    zz[xinx,yinx]   = val
            
#            vmin    = 0
#            vmax    = 60.
            vmin    = None
            vmax    = None
            cmap    = plt.cm.jet

            pcoll   = ax.contourf(xb[:-1],yb[:-1],zz.T,15,vmin=vmin,vmax=vmax,cmap=cmap)
    #        CS      = ax.contour(xb[:-1],yb[:-1],zz.T,15, linewidths=0.5, colors='k')
    #        pcoll   = ax.pcolormesh(xb,yb,zz.T,vmin=vmin,vmax=vmax,cmap=cmap)
    #        ax.scatter(xx,yy,color='0.8',alpha=0.5)

            cbar    = plt.colorbar(pcoll,ax=ax,pad=cbar_pad)
            cbar.set_label('Elevation Angle [deg]',fontdict=fontdict,labelpad=labelpad)


    if fit is not None:
        xx  = fit.index.tolist()
        yy  = fit.fit.tolist()
        ax.plot(xx,yy,lw=6,ls='-',color='yellow')
        
    if obsc_df is not None:
        ax2 = ax.twinx()
        xx  = obsc_df.index.tolist()
        yy  = obsc_df.obsc.tolist()
        ax2.plot(xx,yy,lw=obsc_lw,ls='--',color='white')
        ax2.set_ylim(1.0,0)
        ax2.set_ylabel('Eclipse Obsuration',fontdict=fontdict,labelpad=labelpad)

    if solar_df is not None:
        ax2 = ax.twinx()
        xx  = solar_df.index.tolist()
        yy  = solar_df.el.tolist()
        ax2.plot(xx,yy,lw=4,ls=':',color='white')
        ax2.set_ylim(90,0)
        ax2.set_ylabel('Solar Zenith Angle',fontdict=fontdict,labelpad=labelpad)
        if obsc_df is not None:
            ax2.spines['right'].set_position(('axes', sza_pad))
#            ax2.spines['right'].set_position(('axes', 1.20))

    if legend:
        ax.legend(loc='upper right')
    ax.set_ylabel(ylabel)
    ax.set_ylim(ylim)

def ds_autodate(data_sets,ds_key):
    date    = data_sets[ds_key].get('date','auto')
    if date == 'auto':
        df  = data_sets[ds_key]['df']
        dt_0    = df['datetime'].min()
        dt_1    = df['datetime'].max()
        dt_mid  = (dt_1-dt_0)/2 + dt_0
        date    = datetime.datetime(dt_mid.year,dt_mid.month,dt_mid.day)
    return date

def plot_rx(data_sets,call_0,lat_0,lon_0,smoothing=0,ds_key='eclipse',
        obsc_df=None,mid_df=None,xparam='ut_hrs',**kwargs):

    date        = ds_autodate(data_sets,ds_key)

    xprm        = prmd[xparam]
    xkey        = xprm.get('key',xparam)
    xlim        = xprm.get('lim')
    xlabel      = xprm.get('label')
    xvals       = xprm.get('bins')
    bin_hrs     = xprm.get('bin_size')

    # Calculate Solar Zenith Angle
    if date is not None:
        datetimes   = [date + datetime.timedelta(hours=x) for x in xvals]
    else:
        datetimes   = [datetime.datetime(2017,8,21) + datetime.timedelta(hours=x) for x in xvals]

    azs, els    = sunAzEl(datetimes,lat_0,lon_0)
    solar_df    = pd.DataFrame({'datetimes':datetimes,'az':azs,'el':els},index=xvals)
     
    # Calculate All Parameters
    df_dct          = OrderedDict()
    dct             = data_sets[ds_key]
    df_dct[ds_key]  = OrderedDict()
    for band in bands:
        bdct = OrderedDict()
        df_dct[ds_key][band] = bdct

        df  = dct['df']
        tf  = np.logical_and(df['call_0']==call_0,df['band']==band)
        df  = df[tf].copy()
        bdct['df']  = df

        nvals       = []
        srpt_0      = []
        R_gc        = []
        srpt_0_1st  = []
        R_gc_1st    = []

        for xval in xvals:
            tf  = np.logical_and(df.ut_hrs >= xval,
                                 df.ut_hrs <  xval+bin_hrs)
            df_tmp  = df[tf]
            nvals.append(len(df_tmp))
            srpt_0.append(df_tmp.srpt_0.quantile(0.99))
            R_gc.append(df_tmp.R_gc.quantile(0.99))
            srpt_0_1st.append(df_tmp.srpt_0.quantile(0.10))
            R_gc_1st.append(df_tmp.R_gc.quantile(0.10))

        bdct['nvals']       = nvals
        bdct['srpt_0']      = srpt_0
        bdct['R_gc']        = R_gc
        bdct['srpt_0_1st']  = srpt_0_1st
        bdct['R_gc_1st']    = R_gc_1st
        bdct[xkey]      = xvals

    # Plotting Section ##################### 
    # Create Figure
    fig = plt.figure(figsize=(35,25))
    ny      = len(bands) + 1
    nx      = 3

    ax_hgt  = 1/ny
    col12w  = 0.385
    col3w   = 1.-2*col12w
    vpad    = 0.025
    hpad    = 0.010

    # Plot Map
    tmp_df  = pd.DataFrame({'lat_0':[lat_0],'lon_0':[lon_0]},index=[call_0])
    bottom  = 1 - ax_hgt
    height  = ax_hgt
    left    = 0.
    width   = 0.75*col12w
    ax      = fig.add_axes([left,bottom,width,height])
    df      = data_sets[ds_key]['df']
    plot_map(tmp_df,ax,mid_df=df)

    # Plot Omni
    xprm    = prmd[xkey]
    xlim    = xprm.get('lim')
    xlabel  = xprm.get('label')
    if date is not None:
        sTime   = date + datetime.timedelta(hours=xlim[0])
        eTime   = date + datetime.timedelta(hours=xlim[1]+1)
        omni    = Omni()

        bottom  = 1 - ax_hgt
        height  = ax_hgt-vpad
        left    = 2*col12w
        width   = col3w
        ax      = fig.add_axes([left,bottom,width,height])
        omni.plot_dst_kp(sTime,eTime,ax=ax)
        ax.set_xlim(xlim)
#        ax.set_xlabel(xlabel)

    # Declare Band Subplots
    ax_arr  = OrderedDict()
    for band_inx,band in enumerate(bands[::-1]):
        ax_arr[band] = []
        bottom  = 1 - (band_inx+2)*ax_hgt
        height  = ax_hgt-vpad

        for col in range(nx):
            if col < 2:
                left    = col*col12w
                width   = col12w-hpad
            else:
                left    = col*col12w
                width   = col3w
            ax  = fig.add_axes([left,bottom,width,height])
            ax.set_xlim(xlim)
            if band_inx == len(bands)-1:
                ax.set_xlabel(xlabel)
            ax_arr[band].append(ax)

    lw      = 4
    for band_inx,band in enumerate(bands[::-1]):
        # SNR Plots ############################ 
        ax  = ax_arr[band][0]

        plot_subplot('max_snr',df_dct,band,ax,
                obsc_df=obsc_df,solar_df=solar_df,xparam=xkey)

        band_name   = bandObj.band_dict[band]['name']
        ax.text(-0.125,0.5,band_name,transform=ax.transAxes,
                fontdict={'size':32,'weight':'bold'},rotation=90.,va='center')

        # R_gd Plots ########################### 
        ax  = ax_arr[band][1]
        plot_subplot('max_rgc',df_dct,band,ax,
                obsc_df=obsc_df,solar_df=solar_df,xparam=xkey)

        # Number Plots ######################### 
        ax  = ax_arr[band][2]
        plot_subplot('nvals',df_dct,band,ax,xparam=xkey)

    title   = []
    title.append('{!s}-{!s}'.format(call_0,ds_key))
    if date is not None:
        day     = date.strftime('%A')
        title.append(day)
    fig.text(0.5,0.95,'\n'.join(title),fontdict={'weight':'bold','size':32},ha='center')
    return fig

def plot_rx_sea(data_sets,lat_0=None,lon_0=None,latlon0s=None,smoothing=0,
        ds_key='eclipse',obsc_df=None,xparam='epoch',chain=None,mid_obsc_lim=None,
        **kwargs):

    date        = ds_autodate(data_sets,ds_key)

    xprm        = prmd[xparam]
    xkey        = xprm.get('key',xparam)
    xlim        = xprm.get('lim')
    xlabel      = xprm.get('label')
    hrs         = xprm.get('bins')
    bin_hrs     = xprm.get('bin_size')

    # Calculate Solar Zenith Angle
    if lat_0 is not None and lon_0 is not None:
        if date is not None:
            datetimes   = [date + datetime.timedelta(hours=x) for x in ut_hrs]
        else:
            datetimes   = [datetime.datetime(2017,8,21) + datetime.timedelta(hours=x) for x in ut_hrs]
        azs, els    = sunAzEl(datetimes,lat_0,lon_0)
        solar_df    = pd.DataFrame({'datetimes':datetimes,'az':azs,'el':els},index=ut_hrs)
    else:
        solar_df = None
     
    # Calculate All Parameters
    dct             = data_sets[ds_key]
    df_dct          = OrderedDict()
    df_dct[ds_key]  = OrderedDict()
    for band in bands:
        bdct = OrderedDict()
        df_dct[ds_key][band] = bdct

        df  = dct['df']
        tf  = df['band']==band
        df  = df[tf].copy()
        bdct['df']  = df

        nvals       = []
        srpt_0      = []
        R_gc        = []
        srpt_0_1st  = []
        R_gc_1st    = []

        for hr in hrs:
            tf  = np.logical_and(df[xkey] >= hr,
                                 df[xkey] <  hr+bin_hrs)
            df_tmp  = df[tf]
            nvals.append(len(df_tmp))
            srpt_0.append(df_tmp.srpt_0.quantile(0.99))
            R_gc.append(df_tmp.R_gc.quantile(0.99))
            srpt_0_1st.append(df_tmp.srpt_0.quantile(0.10))
            R_gc_1st.append(df_tmp.R_gc.quantile(0.10))

        bdct['nvals']       = nvals
        bdct['srpt_0']      = srpt_0
        bdct['R_gc']        = R_gc
        bdct['srpt_0_1st']  = srpt_0_1st
        bdct['R_gc_1st']    = R_gc_1st
        bdct[xkey]        = hrs

    # Plotting Section ##################### 
    # Create Figure
    fig = plt.figure(figsize=(35,25))
    ny      = len(bands) + 1
    nx      = 3

    ax_hgt  = 1/ny
    col12w  = 0.385
    col3w   = 1.-2*col12w
    vpad    = 0.025
    hpad    = 0.010

    # Plot Map
    bottom  = 1 - ax_hgt
    height  = ax_hgt
    left    = 0.
    width   = 0.75*col12w
    ax      = fig.add_axes([left,bottom,width,height])
    if latlon0s is not None:
        call_0s = dct['df']['call_0'].unique()
        
        tf      = latlon0s.index.map(lambda x: x in call_0s).tolist()
        df_ll0s = latlon0s[tf]
        df      = data_sets[ds_key]['df']
        plot_map(df_ll0s,ax,mid_df=df)

    # Declare Band Subplots
    ax_arr  = OrderedDict()
    for band_inx,band in enumerate(bands[::-1]):
        ax_arr[band] = []
        bottom  = 1 - (band_inx+2)*ax_hgt
        height  = ax_hgt-vpad

        for col in range(nx):
            if col < 2:
                left    = col*col12w
                width   = col12w-hpad
            else:
                left    = col*col12w
                width   = col3w
            ax  = fig.add_axes([left,bottom,width,height])
            ax.set_xlim(xlim)
            if band_inx == len(bands)-1:
                ax.set_xlabel(xlabel)
            ax_arr[band].append(ax)

    lw      = 4
    for band_inx,band in enumerate(bands[::-1]):
        # SNR Plots ############################ 
        ax  = ax_arr[band][0]

        plot_subplot('max_snr',df_dct,band,ax,
                obsc_df=obsc_df,solar_df=solar_df,xparam=xkey)

        band_name   = bandObj.band_dict[band]['name']
        ax.text(-0.125,0.5,band_name,transform=ax.transAxes,
                fontdict={'size':32,'weight':'bold'},rotation=90.,va='center')

        # R_gd Plots ########################### 
        ax  = ax_arr[band][1]
        plot_subplot('max_rgc',df_dct,band,ax,
                obsc_df=obsc_df,solar_df=solar_df,xparam=xkey)

        # Number Plots ######################### 
        ax  = ax_arr[band][2]
        plot_subplot('nvals',df_dct,band,ax,xparam=xkey)

    title   = []
    if chain == 'All':
        title.append('{!s}'.format(ds_key))
    else:
        title.append('{!s}-{!s}'.format(chain,ds_key))

    if date is not None:
        day     = date.strftime('%A')
        title.append(day)
    if mid_obsc_lim is not None:
#        txt = 'Midpoint Obsc. Limit: {!s}'.format(lim_formatter(mid_obsc_lim))
        txt = 'Max Obscuration {!s}'.format(lim_formatter(mid_obsc_lim))
        title.append(txt)
    fig.text(0.5,0.95,'\n'.join(title),fontdict={'weight':'bold','size':32},ha='center')
    return fig

def plot_rx_sea_publication(data_sets,lat_0=None,lon_0=None,latlon0s=None,latlon1s=None,smoothing=0,
        ds_key='eclipse',obsc_df=None,xparam='epoch',chain=None,mid_obsc_lim=None,include_map=True,
        plot_title=True,plot_alphas=False,param='max_rgc',save_big_map=True,**kwargs):

    xprm        = prmd[xparam]
    xkey        = xprm.get('key',xparam)
    xlim        = xprm.get('lim')
    xlabel      = xprm.get('label')
    hrs         = xprm.get('bins')
    bin_hrs     = xprm.get('bin_size')

    yprm        = prmd[param]
    ykey        = yprm.get('key',param)
    ylim        = yprm.get('lim')
    ybin_size   = yprm.get('bin_size')

    bands       = [1,3,7,14]

    if plot_alphas:
        alphas      = ['a','b','c','d','e','f','g']
    else:
        alphas      = None

#    # Trim the DataFrame to the chosen x and y limits.
    df      = data_sets[ds_key]['df']
    xl      = [xlim[0]-bin_hrs,xlim[1]+bin_hrs] # Needed to make contour compute to the edge.
    tf_x    = lim_tf(df[xkey],xl)

    yl      = [ylim[0]-ybin_size,ylim[1]+ybin_size]
    tf_y    = lim_tf(df[ykey],yl)
    tf      = np.logical_and(tf_x,tf_y)
    df      = df[tf].copy()

    # Only include bands that are being plotted.
    tf      = df.band.map(lambda x: x in bands).tolist()
    df      = df[tf].copy()

    # Compute Eclipse Obscuration Function
    lat_0           =  40.
    lon_0           = -100.
    obsc_df         = calc_obsc_curve(lat_0,lon_0)
    argmax          = obsc_df.obsc.idxmax()
    obsc_df.index   = obsc_df.index - argmax

    date            = ds_autodate(data_sets,ds_key)

    # Calculate Solar Zenith Angle
    solar_df = None
#    if lat_0 is not None and lon_0 is not None:
#        datetimes   = [date + datetime.timedelta(hours=(argmax+x)) for x in hrs]
#        azs, els    = sunAzEl(datetimes,lat_0,lon_0)
#        solar_df    = pd.DataFrame({'datetimes':datetimes,'az':azs,'el':els},index=hrs)
#        obsc_df = None

#    if ds_key == 'control':
#        date    = None
#        day     = None
#    elif ds_key == 'eclipse' or ds_key =='simulation_eclipse':
#        date    = datetime.datetime(2017,8,21)
#    else:
#        date    = datetime.datetime.strptime(ds_key,'%Y%m%d')
     
    # Make sure there is data.
    if len(df) == 0:
        fig = plt.figure()
        fig.text(0.5,0.5,'No Data')
        return fig

    # Split into bands.
    df_dct          = OrderedDict()
    df_dct[ds_key]  = OrderedDict()
    for band in bands:
        bdct = OrderedDict()
        df_dct[ds_key][band] = bdct
        bdct['df']  = df[df['band']==band].copy()

    # Plotting Section ##################### 
    # Create Figure
    fscale  = 1.15
    fig     = plt.figure(figsize=(fscale*25,fscale*12.5))
    ny      = len(bands) + 1
    nx      = 1

    ax_hgt  = 1/ny
    col12w  = 0.385
    col3w   = 1.-2*col12w
    vpad    = 0.035
    hpad    = 0.010

    # Plot Map
    bottom  = 1 - ax_hgt
    height  = ax_hgt
    left    = 0.
    width   = 0.75*col12w
    if latlon0s is not None:
        call_0s = df['call_0'].unique()
        tf      = latlon0s.index.map(lambda x: x in call_0s).tolist()
        df_ll0s = latlon0s[tf]

        if include_map:
            ax      = fig.add_axes([left,bottom,width,height])
            plot_map(df_ll0s,ax,mid_df=df,pad=0.1,rbn_s=250)
            if alphas is not None:
                ax.text(-0.110,0.925,'({!s})'.format(alphas[0]),transform=ax.transAxes,
                        fontdict={'size':26,'weight':'bold'},va='center')
        if save_big_map:
            out_file    = kwargs.get('out_file','big_map')
            tag         = '_{!s}_map'.format(param)
            map_out     = out_file.replace('.png',tag)

            bmap_scl    = 0.75 
            fig_bmap    = plt.figure(figsize=(0.75*30,0.75*10))
            ax_bmap     = fig_bmap.add_subplot(1,1,1)
            plot_map(df_ll0s,ax_bmap,mid_df=df,pad=0.05,rbn_s=400,tx_s=50)
#            fig_bmap.savefig(map_out+'.pdf',bbox_inches='tight')
            fig_bmap.savefig(map_out+'.png',bbox_inches='tight')
            print('Saving: {!s}'.format(map_out+'.png'))
            plt.close(fig_bmap)

            # List RXs for CSV.
            all_call_0s = df.call_0.unique()
            tmp_lst = []
            for rinx,row in df_ll0s.iterrows():
                if rinx in all_call_0s:
                    tmp_lst.append(row)
            tmp_df  = pd.DataFrame(tmp_lst)
            tmp_df.to_csv(map_out+'.call_0.csv')

    if latlon1s is not None:
        if save_big_map:
            call_1s = df['call_1'].unique()
            tf      = latlon1s.index.map(lambda x: x in call_1s).tolist()
            df_ll1s = latlon1s[tf]

            # List TXs for CSV.
            all_call_1s = df.call_1.unique()
            tmp_lst = []
            for rinx,row in df_ll1s.iterrows():
                if rinx in all_call_1s:
                    tmp_lst.append(row)
            tmp_df  = pd.DataFrame(tmp_lst).drop_duplicates(['lat_1', 'lon_1'])
            tmp_df.to_csv(map_out+'.call_1.csv')

    # Declare Band Subplots
    ax_arr  = OrderedDict()
    for band_inx,band in enumerate(bands[::-1]):
        ax_arr[band] = []
        bottom  = 1 - (band_inx+2)*ax_hgt
        height  = ax_hgt-vpad

        for col in range(nx):
            if col < 2:
                left    = col*col12w
                width   = col12w-hpad
            else:
                left    = col*col12w
                width   = col3w
            ax  = fig.add_axes([left,bottom,width,height])
            ax.set_xlim(xlim)
            if band_inx == len(bands)-1:
                ax.set_xlabel(xlabel)
            ax_arr[band].append(ax)

    lw      = 4
    for band_inx,band in enumerate(bands[::-1]):
        out_file    = kwargs.get('out_file')

        # Output CSV File of Data Contained in Plot
        if out_file is not None:
            tag     = '_{!s}MHz_{!s}.csv'.format(band,param)
            csv_out = out_file.replace('.png',tag)
            df      = df_dct[ds_key][band]['df']
            df.to_csv(csv_out,index=False)

        ax          = ax_arr[band][0]
        cbar_pad    = 0.120
#        cbar_pad    = 0.200
        plot_subplot(param,df_dct,band,ax,
                obsc_df=obsc_df,solar_df=solar_df,xparam=xkey,
                obsc_lw=3,legend=False,cbar_pad=cbar_pad,sza_pad=1.1) #1.17 for solar_df

#        band_name   = bandObj.band_dict[band]['name']
        if band == 1:
            band_name = '1.8 MHz'
        elif band == 3:
            band_name = '3.5 MHz'
        elif band == 7:
            band_name = '7 MHz'
        elif band == 14:
            band_name = '14 MHz'
        elif band == 21:
            band_name = '21 MHz'
        elif band == 28:
            band_name = '28 MHz'

        ax.text(-0.185,0.5,band_name,transform=ax.transAxes,
                fontdict={'size':26,'weight':'bold'},rotation=90.,va='center')

        if alphas is not None:
            alpha_inx = band_inx
            if include_map: alpha_inx += 1
            ax.text(-0.185,0.98,'({!s})'.format(alphas[alpha_inx]),transform=ax.transAxes,
                    fontdict={'size':26,'weight':'bold'},va='center')

    if plot_title:
        title   = []
        if chain == 'All':
            title.append('{!s}'.format(ds_key))
        else:
            title.append('{!s}-{!s}'.format(chain,ds_key))
        if mid_obsc_lim is not None:
            txt = 'Midpoint Obsc. Limit: {!s}'.format(lim_formatter(mid_obsc_lim))
            title.append(txt)
        ax  = ax_arr[bands[-1]][0]
        if include_map:
            ypos = 2.50
        else:
            ypos = 1.00
        ax.set_title('\n'.join(title),fontdict={'weight':'bold','size':32},y=ypos)
    return fig

def plot_map(df_ll0s,ax,us=True,mid_df=None,rbn_s=500,tx_s=10,pad=0.05,link_df=None):
    date_0      = datetime.datetime(2017,8,21,14)
    date_1      = datetime.datetime(2017,8,21,22)

    map_prm = {}
    if us:
        map_prm['llcrnrlon'] = -130.
        map_prm['llcrnrlat'] = 20
        map_prm['urcrnrlon'] = -60.
        map_prm['urcrnrlat'] = 55.
    else:
        map_prm['llcrnrlon'] = -180.
        map_prm['llcrnrlat'] = -90
        map_prm['urcrnrlon'] = 180.
        map_prm['urcrnrlat'] = 90.

    hmap    = seqp.maps.HamMap(date_0,date_1,ax,show_title=False,**map_prm)
    hmap.overlay_gridsquares(label_precision=0,major_style={'color':'0.8','dashes':[1,1]})

    lat_0   = df_ll0s['lat_0']
    lon_0   = df_ll0s['lon_0']

    label   = 'RBN Receiver (N = {!s})'.format(len(lat_0))
    hmap.m.scatter(lon_0,lat_0,marker='*',s=rbn_s,color='blue',zorder=50,label=label)

    if mid_df is not None:
        lats    = mid_df['lat_mid']
        lons    = mid_df['lon_mid']
        obsc    = mid_df['obs_max_mid']
        label   = 'Path Midpoint (N = {!s})'.format(len(lats))
        pcoll   = hmap.m.scatter(lons,lats,c=obsc,marker='.',s=tx_s,
                cmap=mpl.cm.jet,vmin=0,vmax=1,label=label,rasterized=True)
        cbar    = plt.colorbar(pcoll,ax=hmap.ax,pad=pad)
        fontdict = {'size':'xx-large','weight':'normal'}
        cbar.set_label('Obscuration',fontdict=fontdict)

        lats    = mid_df['lat_1']
        lons    = mid_df['lon_1']
        latlons = list(set(zip(lats.tolist(),lons.tolist())))
        lats,lons = zip(*latlons)
        label   = 'Transmitter (N = {!s})'.format(len(lats))
        hmap.m.scatter(lons,lats,color='black',marker='.',s=tx_s/2,label=label,zorder=50)

    if link_df is not None:
        lat_0   = float(link_df['lat_0'])
        lon_0   = float(link_df['lon_0'])
        hmap.m.scatter(lon_0,lat_0,marker='*',s=rbn_s,color='None',zorder=100,
                        linewidths=2,edgecolors='k')

        lat_1   = float(link_df['lat_1'])
        lon_1   = float(link_df['lon_1'])
        hmap.m.scatter(lon_1,lat_1,marker='o',s=tx_s,color='None',zorder=100,
                        linewidths=2,edgecolors='k')

        hmap.m.drawgreatcircle(lon_0,lat_0,lon_1,lat_1,ls='--',lw=3,color='k',zorder=100)

#        lat_mid   = float(link_df['lat_mid'])
#        lon_mid   = float(link_df['lon_mid'])
#        hmap.m.scatter(lon_mid,lat_mid,marker='X',s=200,color='white',zorder=100,
#                        linewidths=2,edgecolors='k')
        


    hmap.ax.legend(loc='lower left')

def load_data(ds_keys=None,split_control=False,reset_cache=True):
    if reset_cache:
        prep_output({0:cache_dir},clear=True,php=False)

    # Define and load in Eclipse and Control datasets.
    print('Loading data from CSV files...')

    if ds_keys is None:
        ds_load = sources.keys()
    else:
        ds_load = ds_keys

    data_sets   = {}
    for data_set in ds_load:
        dsd                 = sources.get(data_set)
        data_sets[data_set] = dsd

        print('load_data(): {!s}'.format(data_set))
        if not os.path.exists(dsd['cache_csv']):
            print('Source File: {!s}'.format(dsd['csv_in']))
            df          = pd.read_csv(dsd['csv_in'],index_col=False,parse_dates=[0])
            
            # Reduce number of columns.
            keys = []
            keys.append('datetime')
            keys.append('frequency')
            keys.append('band')
            keys.append('mode')
            keys.append('R_gc')
            keys.append('azm')
            keys.append('call_0')
            keys.append('lat_0')
            keys.append('lon_0')
            keys.append('srpt_0')
            keys.append('call_1')
            keys.append('lat_1')
            keys.append('lon_1')
            keys.append('srpt_1')
            keys.append('lat_mid')
            keys.append('lon_mid')
            keys.append('obs_0_300km')
            keys.append('obs_1_300km')
            keys.append('obs_mid_300km')

            if 'apogee' in df.keys():
                keys.append('plasma_freq_at_apogee')
                keys.append('apogee')
                keys.append('virtual_height')
                keys.append('initial_elev')

#            keys.append('grid_0')
#            keys.append('grid_src_0')
#            keys.append('grid_1')
#            keys.append('grid_src_1')
#            keys.append('source')

    #        keys.append('N_hops')
    #        keys.append('R_gc_mean')
    #        keys.append('lat_mhop_0')
    #        keys.append('lon_mhop_0')
    #        keys.append('lat_mhop_1')
    #        keys.append('lon_mhop_1')
    #        keys.append('lat_mhop_2')
    #        keys.append('lon_mhop_2')
    #        keys.append('lat_mhop_3')
    #        keys.append('lon_mhop_3')
    #        keys.append('lat_mhop_4')
    #        keys.append('lon_mhop_4')
    #        keys.append('km2max_0_300km')
    #        keys.append('km2max_1_300km')
    #        keys.append('km2max_mid_300km')
    #        keys.append('obs_mhop_0_300km')
    #        keys.append('km2max_mhop_0_300km')
    #        keys.append('obs_mhop_1_300km')
    #        keys.append('km2max_mhop_1_300km')
    #        keys.append('obs_mhop_2_300km')
    #        keys.append('km2max_mhop_2_300km')
    #        keys.append('obs_mhop_3_300km')
    #        keys.append('km2max_mhop_3_300km')
    #        keys.append('obs_mhop_4_300km')
    #        keys.append('km2max_mhop_4_300km')
            df  = df[keys].copy()

            # Select only those QSOs that have endpoints in the US.
            lon_min     = -130.
            lon_max     = -60.
            lat_min     =  20.
            lat_max     =  55.

            tf_lat_0    = np.logical_and(df.lat_0 >= lat_min, df.lat_0 < lat_max)
            tf_lon_0    = np.logical_and(df.lon_0 >= lon_min, df.lon_0 < lon_max)
            tf_0        = np.logical_and(tf_lat_0,tf_lon_0)
            tf          = tf_0

            tf_lat_1    = np.logical_and(df.lat_1 >= lat_min, df.lat_1 < lat_max)
            tf_lon_1    = np.logical_and(df.lon_1 >= lon_min, df.lon_1 < lon_max)
            tf_1        = np.logical_and(tf_lat_1,tf_lon_1)
            tf          = np.logical_and(tf_0,tf_1)

            df          = df[tf]

            # Only keep paths R_gc < 3000 km
            tf  = df['R_gc'] < 3000
            df  = df[tf].copy()

            # Calculate Solar Local Time and UT Hours
            ut_hrs          = df.datetime.map(dt2decimalhours)
            df['ut_hrs']    = ut_hrs
            df['slt_mid']   = (ut_hrs + df.lon_mid/15.) % 24.

            # Calculation Obscuration Maxes and Times
    #        sfxs    = ['0','1','mid']
            sfxs    = ['mid']
            for sfx in sfxs:
                latlons = []
                for rinx,row in df.iterrows():
                    lat         = row['lat_'+sfx]
                    lon         = row['lon_'+sfx]
                    latlons.append((lat,lon))

                print('Getting EclipseTimes for {!s}'.format(sfx))
                with mp.Pool() as pool:
                    ecl_times   = list(tqdm.tqdm(pool.imap(get_eclipse_times_tuple,latlons),total=len(latlons)))

#                ecl_times   = []
#                for latlon in latlons:
#                    ecl_time    = get_eclipse_times(*latlon)
#                    ecl_times.append(ecl_time)

                obs_max     = []
                obs_max_dt  = []
                for ecl_time in ecl_times:
                    mx          = ecl_time.get('ecl_maxtime_obs')
                    obs_dt      = ecl_time.get('ecl_maxtime')

                    obs_max.append(mx)
                    obs_max_dt.append(obs_dt)

                df['obs_max_'+sfx]      = obs_max
                df['obs_max_dt_'+sfx]   = obs_max_dt

            df              = df.sort_values('frequency')
            df              = df.sort_values('datetime')
            df.to_csv(dsd['cache_csv'],index=False,compression='bz2')
        else:
            print('Cached File: {!s}'.format(dsd['cache_csv']))
            df = pd.read_csv(dsd['cache_csv'],parse_dates=['datetime','obs_max_dt_mid'])

        df  = tx_on_land(df)
        
        df['call_0']    = df['call_0'].map(clean_call)
        df['call_1']    = df['call_1'].map(clean_call)
        dsd['df']       = df

    # Reject Wednesdays ####################
    if 'control' in data_sets.keys():
        rejects  = []
        rejects.append(datetime.datetime(2017,8,9))
        rejects.append(datetime.datetime(2017,8,16))
        rejects.append(datetime.datetime(2017,8,23))
        rejects.append(datetime.datetime(2017,8,30))
        for reject in rejects:
            df  = data_sets['control']['df']
            tf  = np.logical_or(df.datetime < reject, df.datetime >= reject+datetime.timedelta(days=1))
            data_sets['control']['df'] = df[tf].copy()

    # Split Control Day into Components ####
    if split_control:
        df      = data_sets['control']['df']
        dates   = df.datetime.map(lambda x:x.date()).unique().tolist()
        dates.sort()
        for date in dates:
            tf  = np.logical_and(df.datetime >= date,
                                 df.datetime <  date+datetime.timedelta(days=1))
            df_tmp  = df[tf].copy()

            key = date.strftime('%Y%m%d')
            data_sets[key] = {}
            data_sets[key]['df']        = df_tmp
            data_sets[key]['csv_in']    = data_sets['control']['csv_in']
    return data_sets

def call_0_data(data_sets,call_0):
    counts      = 0
    call_0_ds   = OrderedDict()
    for data_set,dsd in data_sets.items():
        call_0_ds[data_set] = {}
        call_0_ds[data_set]['csv_in']   = dsd.get('csv_in')
        tf  = dsd['df']['call_0'] == call_0
        counts  += np.count_nonzero(tf)
        call_0_ds[data_set]['df'] = dsd['df'][tf].copy()

    if counts == 0:
        call_0_ds = None

    return call_0_ds

def plot_all_rx(df_ll0s,call_0s=None,out_dir=None,fname='000_rx_map.png',**kwargs):
    if out_dir is None:
        out_dir = base_dir

    if call_0s is not None:
        tf      = latlon0s.index.map(lambda x: x in call_0s).tolist()
        df_ll0s = df_ll0s[tf]

    # Plot 1 big maps of all RBN RX's.
#    fig = plt.figure(figsize=(20,16))
    fig = plt.figure(figsize=(0.75*30,0.75*10))
    ax  = fig.add_subplot(1,1,1)
    plot_map(df_ll0s,ax,us=True,**kwargs)
    fpath   = os.path.join(out_dir,fname)
    fig.savefig(fpath,bbox_inches='tight')
    print('Saving: {!s}'.format(fpath))
    plt.close(fig)

#def plot_figure_1(df_ll0s,call_0s=None,out_dir=None,fname='000_rx_map.png',**kwargs):
#    if out_dir is None:
#        out_dir = base_dir
#
#    if call_0s is not None:
#        tf      = latlon0s.index.map(lambda x: x in call_0s).tolist()
#        df_ll0s = df_ll0s[tf]
#
#
#    # Plot 1 big maps of all RBN RX's.
#    fig = plt.figure(figsize=(22.5,7.5))
##    ax  = fig.add_subplot(1,1,1)
#
#    left    = 0.05
#    width   = 1 - left
#    bottom  = 0.5
#    height  = 0.45
#    ax      = fig.add_axes([left,bottom,width,height])
#    plot_map(df_ll0s,ax,us=True,**kwargs)
#
#    left    = 0.05
#    width   = 1 - left
#    bottom  = 0.
#    height  = 0.45
#    ax      = fig.add_axes([left,bottom,width,height])
##    ax.plot(np.arange(100)**2)
#    png     = '2017-08-21 18:18:00-KG5PFD-K9IMM_7.03_eclipse.png'
#    image   = plt.imread(png)
#    ax.imshow(image)
#    ax.axis('off')
#
#    fpath   = os.path.join(out_dir,fname)
#    fig.savefig(fpath,bbox_inches='tight')
#    print('Saving: {!s}'.format(fpath))
#    plt.close(fig)


def chains_from_data_sets(data_sets):
    chains  = OrderedDict()

    # All Stations
    call_0s         = []
    for data_set,dsd in data_sets.items():
        df  = dsd['df']
        call_0s     += df['call_0'].tolist()

    call_0s = list(set(call_0s))
    call_0s.sort()
    chains['All']   = call_0s

#    # Custom Chains ########################
#    call_0s = []
#    call_0s.append('KM3T-2')
#    call_0s.append('W3UA')
#    call_0s.append('K1TTT')
#    call_0s.append('N2GZ')
#    call_0s.append('KQ2Z')
#    call_0s.append('N4ZR/3')
#    call_0s.append('NN3RP')
#    call_0s.append('W4KKN')
#    call_0s.append('WA4RTS1')
#    call_0s.append('AA4VV')
#    call_0s.append('W3OA')
#    call_0s.append('KS4XQ')
#    call_0s.append('KM4LQP')
#    call_0s.append('VE2AED')
#    call_0s.append('VE2WU')
#    call_0s.append('W2LB')
#    call_0s.append('W4AX')
#    chains['East'] = call_0s
#
#    call_0s = []
#    call_0s.append('K2PO')
#    call_0s.append('KF4FIC')
#    call_0s.append('KO7SS')
#    call_0s.append('KU7T')
#    call_0s.append('N7TR')
#    call_0s.append('NC7J')
#    call_0s.append('VE6AO')
#    call_0s.append('VE6JY')
#    call_0s.append('VE6WZ')
#    call_0s.append('VE7CC')
#    call_0s.append('W7HR')
#    call_0s.append('WA7LNW')
#    chains['West'] = call_0s
#
#    call_0s = []
#    call_0s.append('K5JBT')
#    call_0s.append('K8ND')
#    call_0s.append('K9IMM')
#    call_0s.append('N5ME')
#    call_0s.append('N9YKE')
#    call_0s.append('WA9VEE')
#    call_0s.append('WE9V')
#    chains['Central'] = call_0s
    return chains

def get_latlon0s(data_sets):

    all_df  = pd.DataFrame()
    for data_set,dsd in data_sets.items():
        df  = dsd['df']
        all_df  = all_df.append(df,ignore_index=True)
    
    call_0s = all_df['call_0'].unique().tolist()
    call_0s.sort()
    
    lat_0s  = []
    lon_0s  = []
    for call_0 in call_0s:
        df_tmp  = all_df[all_df['call_0']==call_0].copy()
        lats  = df_tmp['lat_0'].unique()
        lons  = df_tmp['lon_0'].unique()
        if lats.std() <= 0.50:
            lat_0   = lats.mean()
        else:
            raise Exception('lat_0 out of tolerance')

        if lons.std() <= 0.50:
            lon_0   = lons.mean()
        else:
            raise Exception('lon_0 out of tolerance')

        lat_0s.append(lat_0)
        lon_0s.append(lon_0)
    latlon0s    = pd.DataFrame({'lat_0':lat_0s,'lon_0':lon_0s},index=call_0s) 
    return latlon0s

def get_latlon1s(data_sets):

    all_df  = pd.DataFrame()
    for data_set,dsd in data_sets.items():
        df  = dsd['df']
        all_df  = all_df.append(df,ignore_index=True)
    
    call_1s = all_df['call_1'].unique().tolist()
    call_1s.sort()
    
    lat_1s  = []
    lon_1s  = []
    for call_1 in call_1s:
        df_tmp  = all_df[all_df['call_1']==call_1].copy()
        lats  = df_tmp['lat_1'].unique()
        lons  = df_tmp['lon_1'].unique()
        if lats.std() <= 0.50:
            lat_1   = lats.mean()
        else:
            raise Exception('lat_1 out of tolerance')

        if lons.std() <= 0.50:
            lon_1   = lons.mean()
        else:
            raise Exception('lon_1 out of tolerance')

        lat_1s.append(lat_1)
        lon_1s.append(lon_1)
    latlon1s = pd.DataFrame({'lat_1':lat_1s,'lon_1':lon_1s},index=call_1s) 
    return latlon1s

def get_closest(lat,lon,latlons):
    keys    = latlons.keys()
    for key in keys:
        if 'lat' in key:
            latkey = key
        elif 'lon' in key:
            lonkey = key
    
    lldf    = latlons.copy()
    dists   = []
    for rinx,row in lldf.iterrows():
        rlat    = row[latkey]
        rlon    = row[lonkey]

        dist    = Re*seqp.geopack.greatCircleDist(lat,lon,rlat,rlon)
        dists.append(dist)

    lldf['dist']    = dists
    
    idx             = lldf.dist.idxmin()
    closest         = lldf.loc[idx]

    return closest

class Omni():
    def __init__(self):
        omni_csv        = 'data/omni/omni2_3828.csv'
        omni_df         = pd.read_csv(omni_csv,
                            parse_dates={'datetime':['year','doy','hr']},
                            date_parser=self._date_parser)
        omni_df         = omni_df.set_index('datetime')
        omni_df['Kp']   = omni_df['Kp_x10']/10.
        del omni_df['Kp_x10']
        self.df     = omni_df

    def _date_parser(self,years,doys,hrs):
        datetimes = []
        for year,doy,hr in zip(years,doys,hrs):
            dt  = datetime.datetime(int(year),1,1)+datetime.timedelta(days=(int(doy)-1),hours=int(hr))
            datetimes.append(dt)
        return datetimes

    def plot_dst_kp(self,sTime,eTime,ax,xlabels=True):
        """
        DST and Kp
        """

        tf  = np.logical_and(self.df.index >= sTime, self.df.index < eTime)
        df  = self.df[tf].copy()

        ut_hrs  = [dt.hour + dt.minute/60. + dt.second/3600. for dt in df.index]

        lines       =[]

        xx = ut_hrs
        yy = df['Dst_nT'].tolist()

        tmp,        = ax.plot(xx,yy,label='Dst [nT]',color='k')
#        ax.fill_between(xx,0,yy,color='0.75')
        lines.append(tmp)
        ax.set_ylabel('Dst [nT]')
        ax.axhline(0,color='k',ls='--')
        ax.set_ylim(-200,50)

        # Kp ###################################
        ax_1        = ax.twinx()
#        ax.set_zorder(ax_1.get_zorder()+1)
        ax.patch.set_visible(False)
        low_color   = 'green'
        mid_color   = 'darkorange'
        high_color  = 'red'
        label       = 'Kp'

        xvals       = np.array(ut_hrs)
        kp          = np.array(df['Kp'].tolist())

        if len(kp) > 0:
            color       = low_color
            kp_markersize = 10
            markers,stems,base  = ax_1.stem(xvals,kp,color=color)
            for stem in stems:
                stem.set_color(color)
            markers.set_color(color)
            markers.set_label('Kp Index')
            markers.set_markersize(kp_markersize)
            lines.append(markers)

            tf = np.logical_and(kp >= 4, kp < 5)
            if np.count_nonzero(tf) > 0:
                xx      = xvals[tf]
                yy      = kp[tf]
                color   = mid_color
                markers,stems,base  = ax_1.stem(xx,yy,color=color)
                for stem in stems:
                    stem.set_color(color)
                markers.set_color(color)
                markers.set_markersize(kp_markersize)
                lines.append(markers)

            tf = kp > 5
            if np.count_nonzero(tf) > 0:
                xx      = xvals[tf]
                yy      = kp[tf]
                color   = high_color
                markers,stems,base  = ax_1.stem(xx,yy,color=color)
                for stem in stems:
                    stem.set_color(color)
                markers.set_color(color)
                markers.set_markersize(kp_markersize)
                lines.append(markers)

        ax_1.set_ylabel('Kp Index')
        ax_1.set_ylim(0,9)
        ax_1.set_yticks(np.arange(10))
        for tk,tl in zip(ax_1.get_yticks(),ax_1.get_yticklabels()):
            if tk < 4:
                color = low_color
            elif tk == 4:
                color = mid_color
            else:
                color = high_color
            tl.set_color(color)

        plt.sca(ax)

def generate_station_plots(chains,latlon0s,data_sets):
    for chain,call_0s in chains.items():
        out_dir = os.path.join(base_dir,'stations',chain)
        prep_output({0:out_dir},clear=False,php=php)

        # Plot 1 big map of all RBN RX's.
        plot_all_rx(latlon0s,call_0s,out_dir)

        # Create run list/dictionaries.
        for call_0 in call_0s:
            data_sets_c0 = call_0_data(data_sets,call_0)
            if data_sets_c0 is None: continue

            lat_0       = latlon0s.loc[call_0]['lat_0']
            lon_0       = latlon0s.loc[call_0]['lon_0']
            obsc_df     = calc_obscuration(call_0,lat_0,lon_0)
            for ds_key in data_sets_c0.keys():
                print('Plotting: {!s} - {!s}-{!s}'.format(chain,call_0,ds_key))
                fig         = plot_rx(data_sets_c0,call_0,lat_0,lon_0,ds_key=ds_key,obsc_df=obsc_df)
                ffile       = '{:03.3f}_{!s}-{!s}.png'.format(lat_0,call_0.replace('/','-'),ds_key)
                fpath       = os.path.join(out_dir,ffile)
                fig.savefig(fpath,bbox_inches='tight')
                print('Saving: {!s}'.format(fpath))
                plt.close(fig)

def generate_sea(chains,latlon0s,data_sets,mid_obsc_lim=None,publication=False,latlon1s=None,**kwargs):
    chains_tmp  = chains.copy()
    max_obscs   = []

    sea_dict = OrderedDict()
    for chain,call_0s in chains_tmp.items():
        sea_dict[chain] = OrderedDict()

        # Create run list/dictionaries.
        for ds_key in data_sets.keys():
            sea_dict[chain][ds_key] = OrderedDict()
            sea_df  = pd.DataFrame()
            for call_0 in call_0s:
                data_sets_c0 = call_0_data(data_sets,call_0)
                if data_sets_c0 is None: continue

                lat_0       = latlon0s.loc[call_0]['lat_0']
                lon_0       = latlon0s.loc[call_0]['lon_0']
                obsc_df     = calc_obscuration(call_0,lat_0,lon_0)

                max_obscs.append(obsc_df['obsc'].max())
                df          = data_sets_c0[ds_key]['df']

                obs_ut      = df.obs_max_dt_mid.map(dt2decimalhours)
                df['epoch'] = df['ut_hrs'] - obs_ut
                if mid_obsc_lim is not None:
                    tf  = lim_tf(df['obs_max_mid'],mid_obsc_lim)
                    df  = df[tf].copy()
                    
                sea_df      = sea_df.append(df,ignore_index=True)

            sea_dict[chain][ds_key]['df'] = sea_df
    
    for chain,data_sets_c0 in sea_dict.items():
        for ds_key in data_sets_c0.keys():
            if publication:
                include_map = kwargs.get('include_map',True)
                plot_alphas = kwargs.get('plot_alphas',True)
                plot_title  = kwargs.get('plot_title',True)
                param       = kwargs.get('param','max_rgc')
                out_dir = os.path.join(base_dir,'sea_publication',ds_key,chain,param)
            else:
                out_dir = os.path.join(base_dir,'sea',ds_key,chain)
            prep_output({0:out_dir},clear=False)

            ffile       = 'SEA-{!s}-{!s}.png'.format(chain,ds_key)
            if mid_obsc_lim is not None:
                ffile       = 'SEA-{!s}-{!s}_MOL{!s}-{!s}.png'.format(chain,ds_key,mid_obsc_lim[0],mid_obsc_lim[1])

            if publication:
                ffile   = 'PUB-{!s}'.format(ffile)
            fpath       = os.path.join(out_dir,ffile)

            df      = data_sets_c0[ds_key]['df']
            call_0s = df.call_0.unique()

            print('Plotting Superposed Epoch Analysis: {!s}-{!s}'.format(chain,ds_key))
            if publication:
                fig         = plot_rx_sea_publication(data_sets_c0,ds_key=ds_key,latlon0s=latlon0s,chain=chain,mid_obsc_lim=mid_obsc_lim,
                                    include_map=include_map,plot_alphas=plot_alphas,plot_title=plot_title,param=param,out_file=fpath,latlon1s=latlon1s)
            else:
                fig         = plot_rx_sea(data_sets_c0,ds_key=ds_key,latlon0s=latlon0s,chain=chain,mid_obsc_lim=mid_obsc_lim)

            print('Saving: {!s}'.format(fpath))
            fig.savefig(fpath,bbox_inches='tight')

            plt.close(fig)

def compare_sims(data_sets):
    ds_0   = 'simulation_eclipse'
    ds_1   = 'Eclipse Simulation'

    if ds_0 not in data_sets.keys() or ds_1 not in data_sets.keys():
        return data_sets

    df_0    = data_sets[ds_0]['df']
    df_1    = data_sets[ds_1]['df']

    keys    = []
    keys.append('datetime')
    keys.append('frequency')
    keys.append('band')
    keys.append('mode')
    keys.append('R_gc')
    keys.append('azm')
    keys.append('call_0')
    keys.append('lat_0')
    keys.append('lon_0')

    keys.append('srpt_0')
    keys.append('call_1')
    keys.append('lat_1')
    keys.append('lon_1')

#    keys.append('srpt_1')
    keys.append('lat_mid')
    keys.append('lon_mid')
    keys.append('obs_0_300km')
    keys.append('obs_1_300km')
    keys.append('obs_mid_300km')
    keys.append('ut_hrs')
    keys.append('slt_mid')
    keys.append('obs_max_mid')
    keys.append('obs_max_dt_mid')

    df_00   = df_0.set_index(keys)
    df_11   = df_1.set_index(keys)

    oi0     = df_00.index.difference(df_11.index)
    oi1     = df_11.index.difference(df_00.index)
#    union00 = df_00.index.intersection(df_11.index)
    union11 = df_11.index.intersection(df_00.index)

    df_res  = df_00.loc[oi0].reset_index()
    ds              = 'Only in DS0'
    dsd             = {}
    dsd['df']       = df_res
    data_sets[ds]   = dsd

    df_res  = df_11.loc[oi1].reset_index()
    ds              = 'Only in DS1'
    dsd             = {}
    dsd['df']       = df_res
    data_sets[ds]   = dsd

#    df_res  = df_00.loc[union00].reset_index()
#    ds              = 'Union 00'
#    dsd             = {}
#    dsd['df']       = df_res
#    data_sets[ds]   = dsd

    df_res  = df_11.loc[union11].reset_index()
    ds              = 'Union 11'
    dsd             = {}
    dsd['df']       = df_res
    data_sets[ds]   = dsd

#    del data_sets[ds_0]
#    del data_sets[ds_1]
    return data_sets

def west_east(data_sets,ds_key='Eclipse Simulation'):
    
    dsd = data_sets.get(ds_key)
    if dsd is None:
        return data_sets
    
    df  = dsd['df']

    # West
    tf  = df['lon_1'] < -100
    ds              = 'West TX {!s}'.format(ds_key)
    dsd             = {}
    dsd['df']       = df[tf].copy()
    data_sets[ds]   = dsd

    # East
    tf  = df['lon_1'] >= -100
    ds              = 'East TX {!s}'.format(ds_key)
    dsd             = {}
    dsd['df']       = df[tf].copy()
    data_sets[ds]   = dsd

    return data_sets

def apogee_filter(data_sets,ds_key='Eclipse Simulation',altitude=125):
    
    dsd = data_sets.get(ds_key)
    if dsd is None:
        return data_sets
    
    df  = dsd['df']

    tf  = df['apogee'] < 125
    ds              = u'{!s} (h < {!s} km)'.format(ds_key,altitude)
    dsd             = {}
    dsd['df']       = df[tf].copy()
    data_sets[ds]   = dsd

    tf  = df['apogee'] >= 125
    ds              = u'{!s} (h \u2265 {!s} km)'.format(ds_key,altitude)
    dsd             = {}
    dsd['df']       = df[tf].copy()
    data_sets[ds]   = dsd

    return data_sets


def csv_for_supporting(data_sets,ds_key):

    df  = data_sets[ds_key]['df'].copy()

    df['source']    = 'rbn'

    keys = []
    keys.append('datetime')
    keys.append('frequency')
    keys.append('mode')
    keys.append('source')
    keys.append('R_gc')
    keys.append('azm')
    keys.append('call_0')
    keys.append('lat_0')
    keys.append('lon_0')
    keys.append('srpt_0')
    keys.append('call_1')
    keys.append('lat_1')
    keys.append('lon_1')
    keys.append('lat_mid')
    keys.append('lon_mid')
    keys.append('slt_mid')
    keys.append('obs_max_mid')
    keys.append('obs_max_dt_mid')
    df = df[keys].copy()

#    keys.append('obs_0_300km')
#    keys.append('obs_1_300km')
#    keys.append('obs_mid_300km')

#    fname   = '{!s}_pubCsv.csv.bz2'.format(ds_key)
    fname   = 'S2_rbn_eclipse.csv.bz2'
    fpath   = os.path.join(base_dir,fname)
    df.to_csv(fpath,index=False,compression='bz2')
    return fname

if __name__ == '__main__':
    reset_cache     = False
    split_control   = False
    clear_output    = True
    plot_rx_map     = True
    ray_trace       = False
    plot_rti        = False
    plot_sea        = False
    plot_sea_pub    = True

    ds_keys = []
    ds_keys.append('SEQP RBN Observations')
    ds_keys.append('Eclipse Simulation')

    data_sets   = load_data(ds_keys,split_control,reset_cache)

#    csv_for_supporting(data_sets,'SEQP RBN Observations')

    print('Appogee Filter...')
    data_sets   = apogee_filter(data_sets)
##    data_sets   = compare_sims(data_sets)
##    data_sets   = west_east(data_sets)

    print('Finding Unique Lat/Lons...')
    latlon0s    = get_latlon0s(data_sets)
    latlon1s    = get_latlon1s(data_sets)
    chains      = chains_from_data_sets(data_sets)

    prep_output({0:base_dir},clear=clear_output,php=php)

    if plot_rx_map:
        print('Plotting Figure 1 Map...')
        ds = data_sets.get('SEQP RBN Observations')
        map_dct = {}
        if ds is not None:
            # Get dataframe for plotting midpoints and transmitters.
            df      = ds.get('df')
            tf      = df.band.map(lambda x: x in [1,3,7,14,21,28])
            df      = df[tf].copy()

            lat_0,lon_0 = 43.,-90.
            s_0   = get_closest(lat_0,lon_0,latlon0s)

            lat_1,lon_1 = 31.5,-92.
            s_1   = get_closest(lat_1,lon_1,latlon1s)

            dft     = pd.DataFrame([{'call_0':s_0.name,'lat_0':s_0.lat_0,'lon_0':s_0.lon_0,
                                     'call_1':s_1.name,'lat_1':s_1.lat_1,'lon_1':s_1.lon_1}])

            dft     = calculate_Razm(dft)
            dft     = calculate_midpoint(dft)

            ecl_time                = get_eclipse_times(float(dft['lat_mid']),float(dft['lon_mid']))
            dft['obs_max_mid']      = ecl_time.get('ecl_maxtime_obs')
            dft['obs_max_dt_mid']   = ecl_time.get('ecl_maxtime')
            dft['frequency']        = 7.030

            map_dct = dict(mid_df=df,link_df=dft,pad=0.05,rbn_s=400,tx_s=50)
            plot_all_rx(latlon0s,**map_dct)

            if ray_trace:
                import eclipse_sim
                jobs    = generate_sim_jobs(dft)
                for job in jobs:
                    eclipse_sim.run_job(job,eclipse_flags=[1])

    if plot_rti:
        generate_station_plots(chains,latlon0s,data_sets)

    # Superposed Epoch Analysis #################################################### 
    mols    = []
#    mols.append((None,0.90))
    mols.append((0.90,None))

    if plot_sea:
        generate_sea(chains,latlon0s,data_sets)
        for mol in mols:
            generate_sea(chains,latlon0s,data_sets,mid_obsc_lim=mol)

    if plot_sea_pub:
        print('Plotting Superposed Epoch Analysis (SEA) Publication Quality...')
        rd  = {}
        rd['include_map']   = True
        rd['plot_alphas']   = True
        rd['plot_title']    = True
        rd['latlon1s']      = latlon1s

        params  = []
        params.append('max_rgc')
#        params.append('max_snr')
        for param in params:
#            generate_sea(chains,latlon0s,data_sets,publication=True,param=param,**rd)
            for mol in mols:
                generate_sea(chains,latlon0s,data_sets,publication=True,mid_obsc_lim=mol,param=param,**rd)
