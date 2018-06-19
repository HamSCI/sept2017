import shutil,os
import datetime
from collections import OrderedDict

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import cartopy.crs as ccrs
import matplotlib.patches as mpatches

import numpy as np
import pandas as pd

import geopack

de_prop         = {'marker':'^','edgecolor':'k','facecolor':'white'}
dxf_prop        = {'marker':'*','color':'blue'}
dxf_leg_size    = 150
dxf_plot_size   = 50

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

def make_dir(path,clear=False,php=False):
    prep_output({0:path},clear=clear,php=php)

def clear_dir(path,clear=True,php=False):
    prep_output({0:path},clear=clear,php=php)

def prep_output(output_dirs={0:'output'},clear=False,width_100=False,img_extra='',php=False):
    if width_100:
        img_extra = "width='100%'"

    txt = []
    txt.append('<?php')
    txt.append('foreach (glob("*.png") as $filename) {')
    txt.append('    echo "<img src=\'$filename\' {img_extra}> ";'.format(img_extra=img_extra))
    txt.append('}')
    txt.append('?>')
    show_all_txt = '\n'.join(txt)

    txt = []
    txt.append('<?php')
    txt.append('foreach (glob("*.png") as $filename) {')
    txt.append('    echo "<img src=\'$filename\' {img_extra}> <br />";'.format(img_extra=img_extra))
    txt.append('}')
    txt.append('?>')
    show_all_txt_breaks = '\n'.join(txt)

    for value in output_dirs.values():
        if clear:
            try:
#                shutil.rmtree(value)
                os.system('rm -rf {}/*'.format(value))
            except:
                pass
        try:
            os.makedirs(value)
        except:
            pass
        if php:
            with open(os.path.join(value,'0000-show_all.php'),'w') as file_obj:
                file_obj.write(show_all_txt)
            with open(os.path.join(value,'0000-show_all_breaks.php'),'w') as file_obj:
                file_obj.write(show_all_txt_breaks)

def cc255(color):
    cc = matplotlib.colors.ColorConverter().to_rgb
    trip = np.array(cc(color))*255
    trip = [int(x) for x in trip]
    return tuple(trip)

class BandData(object):
    def __init__(self,cmap='HFRadio',vmin=0.,vmax=30.):
        if cmap == 'HFRadio':
            self.cmap   = self.hf_cmap(vmin=vmin,vmax=vmax)
        else:
            self.cmap   = matplotlib.cm.get_cmap(cmap)

        self.norm   = matplotlib.colors.Normalize(vmin=vmin,vmax=vmax)

        # Set up a dictionary which identifies which bands we want and some plotting attributes for each band
        bands   = []
        bands.append((28.0,  10))
        bands.append((21.0,  15))
        bands.append((14.0,  20))
        bands.append(( 7.0,  40))
        bands.append(( 3.5,  80))
        bands.append(( 1.8, 160))

        self.__gen_band_dict__(bands)

    def __gen_band_dict__(self,bands):
        dct = OrderedDict()
        for freq,meters in bands:
            key = int(freq)
            tmp = {}
            tmp['meters']       = meters
            tmp['name']         = '{!s} m'.format(meters)
            tmp['freq']         = freq
            tmp['freq_name']    = '{:g} MHz'.format(freq)
            tmp['color']        = self.get_rgba(freq)
            dct[key]            = tmp
        self.band_dict          = dct

    def get_rgba(self,freq):
        nrm     = self.norm(freq)
        rgba    = self.cmap(nrm)
        return rgba

    def get_hex(self,freq):

        freq    = np.array(freq)
        shape   = freq.shape
        if shape == ():
            freq.shape = (1,)

        freq    = freq.flatten()
        rgbas   = self.get_rgba(freq)

        hexes   = []
        for rgba in rgbas:
            hexes.append(matplotlib.colors.rgb2hex(rgba))

        hexes   = np.array(hexes)
        hexes.shape = shape
        return hexes

    def hf_cmap(self,name='HFRadio',vmin=0.,vmax=30.):
        fc = {}
        my_cdict = fc
        fc[ 0.0] = (  0,   0,   0)
        fc[ 1.8] = cc255('violet')
        fc[ 3.0] = cc255('blue')
        fc[ 8.0] = cc255('aqua')
        fc[10.0] = cc255('green')
        fc[13.0] = cc255('green')
        fc[17.0] = cc255('yellow')
        fc[21.0] = cc255('orange')
        fc[28.0] = cc255('red')
        fc[30.0] = cc255('red')
        cmap    = cdict_to_cmap(fc,name=name,vmin=vmin,vmax=vmax)
        return cmap

def cdict_to_cmap(cdict,name='CustomCMAP',vmin=0.,vmax=30.):
	norm = matplotlib.colors.Normalize(vmin=vmin,vmax=vmax)
	
	red   = []
	green = []
	blue  = []
	
	keys = list(cdict.keys())
	keys.sort()
	
	for x in keys:
	    r,g,b, = cdict[x]
	    x = norm(x)
	    r = r/255.
	    g = g/255.
	    b = b/255.
	    red.append(   (x, r, r))
	    green.append( (x, g, g))
	    blue.append(  (x, b, b))
	cdict = {'red'   : tuple(red),
		 'green' : tuple(green),
		 'blue'  : tuple(blue)}
	cmap  = matplotlib.colors.LinearSegmentedColormap(name, cdict)
	return cmap

def sun_pos(dt=None):
    """This function computes a rough estimate of the coordinates for
    the point on the surface of the Earth where the Sun is directly
    overhead at the time dt. Precision is down to a few degrees. This
    means that the equinoxes (when the sign of the latitude changes)
    will be off by a few days.

    The function is intended only for visualization. For more precise
    calculations consider for example the PyEphem package.

    Parameters
    ----------
    dt: datetime
        Defaults to datetime.utcnow()

    Returns
    -------
    lat, lng: tuple of floats
        Approximate coordinates of the point where the sun is
        in zenith at the time dt.

    """
    if dt is None:
        dt = datetime.datetime.utcnow()

    axial_tilt = 23.4
    ref_solstice = datetime.datetime(2016, 6, 21, 22, 22)
    days_per_year = 365.2425
    seconds_per_day = 24*60*60.0

    days_since_ref = (dt - ref_solstice).total_seconds()/seconds_per_day
    lat = axial_tilt*np.cos(2*np.pi*days_since_ref/days_per_year)
    sec_since_midnight = (dt - datetime.datetime(dt.year, dt.month, dt.day)).seconds
    lng = -(sec_since_midnight/seconds_per_day - 0.5)*360
    return lat, lng


def fill_dark_side(ax, time=None, *args, **kwargs):
    """
    Plot a fill on the dark side of the planet (without refraction).

    Parameters
    ----------
        ax : Matplotlib axes
            The axes to plot on.
        time : datetime
            The time to calculate terminator for. Defaults to datetime.utcnow()
        **kwargs :
            Passed on to Matplotlib's ax.fill()

    """
    lat, lng = sun_pos(time)
    pole_lng = lng
    if lat > 0:
        pole_lat = -90 + lat
        central_rot_lng = 180
    else:
        pole_lat = 90 + lat
        central_rot_lng = 0

    rotated_pole = ccrs.RotatedPole(pole_latitude=pole_lat,
                                    pole_longitude=pole_lng,
                                    central_rotated_longitude=central_rot_lng)

    x = np.empty(360)
    y = np.empty(360)
    x[:180] = -90
    y[:180] = np.arange(-90, 90.)
    x[180:] = 90
    y[180:] = np.arange(90, -90., -1)

    ax.fill(x, y, transform=rotated_pole, **kwargs)

def band_legend(ax,loc='lower center',markerscale=0.5,prop={'size':10},
        title=None,bbox_to_anchor=None,rbn_rx=True,ncdxf=False,ncol=None,band_data=None):

    if band_data is None:
        band_data = BandData()

    handles = []
    labels  = []

    # Force freqs to go low to high regardless of plotting order.
    band_list   = list(band_data.band_dict.keys())
    band_list.sort()
    for band in band_list:
        color   = band_data.band_dict[band]['color']
        label   = band_data.band_dict[band]['freq_name']

        count   = band_data.band_dict[band].get('count')
        if count is not None:
            label += '\n(n={!s})'.format(count)

        handles.append(mpatches.Patch(color=color,label=label))
        labels.append(label)

    fig_tmp = plt.figure()
    ax_tmp = fig_tmp.add_subplot(111)
    ax_tmp.set_visible(False)
    if rbn_rx:
        scat = ax_tmp.scatter(0,0,s=50,**de_prop)
        labels.append('Receiver')
        handles.append(scat)
    if ncdxf:
        scat = ax_tmp.scatter(0,0,s=dxf_leg_size,**dxf_prop)
        labels.append('NCDXF Beacon')
        handles.append(scat)

    if ncol is None:
        ncol = len(labels)
    
    legend = ax.legend(handles,labels,ncol=ncol,loc=loc,markerscale=markerscale,prop=prop,title=title,bbox_to_anchor=bbox_to_anchor,scatterpoints=1)
    return legend

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

def load_spots_csv(date_str,data_sources=[1,2],loc_sources=['P','Q'],
        rgc_lim=None,filter_region=None,filter_region_kind='mids'):
    """
    Load spots from CSV file and filter for network/location source quality.
    Also provide range and regional filtering, compute midpoints, ut_hrs,
    and slt_mid.

    data_sources: list, i.e. [1,2]
        0: dxcluster
        1: WSPRNet
        2: RBN

    loc_sources: list, i.e. ['P','Q']
        P: user Provided
        Q: QRZ.com or HAMCALL
        E: Estimated using prefix
    """

    CSV_FILE_PATH   = "data/spot_csvs/{}.csv.bz2"
    df              = pd.read_csv(CSV_FILE_PATH.format(date_str),parse_dates=['occurred'])

    # Select spotting networks
    if data_sources is not None:
        tf  = df.source.map(lambda x: x in data_sources)
        df  = df[tf].copy()

    # Filter location source
    if loc_sources is not None:
        tf  = df.tx_loc_source.map(lambda x: x in loc_sources)
        df  = df[tf].copy()

        tf  = df.rx_loc_source.map(lambda x: x in loc_sources)
        df  = df[tf].copy()

    # Path Length Filtering
    if rgc_lim is not None:
        tf  = np.logical_and(df['dist_Km'] >= rgc_lim[0],
                             df['dist_Km'] <  rgc_lim[1])
        df  = df[tf].copy()

#    cols = list(df) + ["md_lat", "md_long"]
#    df = df.reindex(columns=cols)
    midpoints       = geopack.midpoint(df["tx_lat"], df["tx_long"], df["rx_lat"], df["rx_long"])
    df['md_lat']    = midpoints[0]
    df['md_long']   = midpoints[1]

    # Regional Filtering
    if filter_region is not None:
        df      = regional_filter(filter_region,df,kind=filter_region_kind)

    df["ut_hrs"]    = df['occurred'].map(lambda x: x.hour + x.minute/60. + x.second/3600.)
    df['slt_mid']   = (df['ut_hrs'] + df['md_long']/15.) % 24.

    return df

