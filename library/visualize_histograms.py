import os
import glob
import datetime
import dateutil
from collections import OrderedDict
import ast

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import PolyCollection
import cartopy.crs as ccrs

import numpy as np
import pandas as pd
import xarray as xr
import netCDF4

import tqdm

from . import gen_lib as gl
from .timeutils import daterange
from .geospace_env import GeospaceEnv

pdict   = {}

dct = {}
dct['log_z']        = False
pdict['z_score']    = dct

dct = {}
dct['log_z']        = False
pdict['pct_err']    = dct

dct = {}
dct['log_z']            = False
pdict['mean_subtract']  = dct

band_dct = OrderedDict()
dct             = {'label':'28 MHz'}
band_dct[28]    = dct

dct             = {'label':'21 MHz'}
band_dct[21]    = dct

dct             = {'label':'14 MHz'}
band_dct[14]    = dct

dct             = {'label':'7 MHz'}
band_dct[7]     = dct

dct             = {'label':'3.5 MHz'}
band_dct[3]     = dct

dct             = {'label':'1.8 MHz'}
band_dct[1]     = dct

class ncLoader(object):
    def __init__(self,sTime,eTime=None,srcs=None,**kwargs):
        if eTime is None:
            eTime = sTime + datetime.timedelta(hours=24)

        self.sTime      = sTime
        self.eTime      = eTime
        self.srcs       = srcs
        self.kwargs     = kwargs

        self._set_basename()
        self._get_fnames()
        self._load_ncs()

    def _set_basename(self):
        bname   = os.path.basename(self.srcs)
        bname   = bname.strip('*.')
        bname   = bname.strip('.nc')
        self.basename   = bname

    def _get_fnames(self):
        tmp = glob.glob(self.srcs)
        tmp.sort()
        
        dates       = daterange(self.sTime,self.eTime)
        date_strs   = [x.strftime('%Y%m%d') for x in dates]

        fnames      = []
        for fn in tmp:
            bn  = os.path.basename(fn)
            if bn[:8] in date_strs:
                fnames.append(fn)

        self.fnames = fnames
        return fnames
            
    def _load_ncs(self):
        prefixes    = ['map','time_series']

        # Return None if no data to load.
        if self.fnames == []:
            self.maps       = None
            self.datasets   = None
            return

        dss     = {}
        print(' Loading files...')
        for nc_bz2 in self.fnames:
            print(' --> {!s}'.format(nc_bz2))
            mbz2    = gl.MyBz2(nc_bz2)
            mbz2.uncompress()

            nc      = mbz2.unc_name
            # Identify Groups in netCDF File
            with netCDF4.Dataset(nc) as nc_fl:
                groups  = [group for group in nc_fl.groups['time_series'].groups.keys()]

            # Store DataSets (dss) from each group in an OrderedDict()
            for prefix in prefixes:
                if prefix not in dss:
                    dss[prefix] = OrderedDict()

                for group in groups:
                    grp = '/'.join([prefix,group])
                    with xr.open_dataset(nc,group=grp) as fl:
                        ds      = fl.load()

                    # Calculate time vector relative to self.sTime
                    hrs         = np.array(ds.coords[group])
                    dt_0        = pd.Timestamp(np.array(ds['ut_sTime']).tolist())
                    time_vec    = [(dt_0 + pd.Timedelta(hours=x) - self.sTime).total_seconds()/3600. for x in hrs]

                    ds.coords[group]        = time_vec
                    ds.coords['ut_sTime']   = [self.sTime]

                    if prefix == 'map':
                        dt_vec      = np.array([dt_0 + pd.Timedelta(hours=x) for x in hrs])
                        tf          = np.logical_and(dt_vec >= self.sTime, dt_vec < self.eTime)
                        tmp_map_ds  = ds[{group:tf}].sum(group,keep_attrs=True)

                        map_ds      = dss[prefix].get(group)
                        if map_ds is None:
                            map_ds      = tmp_map_ds
                        else:
                            map_attrs = map_ds['spot_density'].attrs
                            map_ds += tmp_map_ds
                            map_ds['spot_density'].attrs = map_attrs
                        dss[prefix][group]  = map_ds
                    else:
                        if group not in dss[prefix]:
                            dss[prefix][group]  = []
                        dss[prefix][group].append(ds)
            mbz2.remove()

        # Concatenate Time Series Data
        xlim        = (0, (self.eTime-self.sTime).total_seconds()/3600.)
        print(' Concatenating data...')
        prefix  = 'time_series'
        xdct    = dss[prefix]
        for group,ds_list in xdct.items():
            ds          = xr.concat(ds_list,group)
            for data_var in ds.data_vars:
                print(prefix,group,data_var)
                attrs   = ds[data_var].attrs
                attrs.update({'xlim':str(xlim)})
                ds[data_var].attrs = attrs
            dss[prefix][group]  = ds

        self.datasets   = dss

    def plot(self,baseout_dir='output',xlim=None,ylim=None,xunits='datetime',subdir=None,
            geospace_env=None,**kwargs):
        if self.datasets is None:
            return

        if geospace_env is None:
            geospace_env    = GeospaceEnv()

#        map_da  = self.maps['spot_density']
        xlim_in = xlim

        for group,ds in self.datasets['time_series'].items():
            map_da  = self.datasets['map'][group]['spot_density']

            outdir  = os.path.join(baseout_dir,group)
            if subdir is not None:
                outdir = os.path.join(outdir,subdir)
            gl.prep_output({0:outdir},clear=False)

            for data_var in ds.data_vars:
                data_da = ds[data_var].copy()

                xlim    = xlim_in
                if xlim is None:
                    xlim = ast.literal_eval(data_da.attrs.get('xlim','None'))

                if ylim is None:
                    ylim = ast.literal_eval(data_da.attrs.get('ylim','None'))

                if xunits == 'datetime':
                    hrs         = np.array(data_da.coords[group])
                    dt_vec      = [self.sTime + pd.Timedelta(hours=x) for x in hrs]
                    data_da.coords[group] = dt_vec

                    if xlim is not None:
                        xlim_0      = pd.Timedelta(hours=xlim[0]) + self.sTime
                        xlim_1      = pd.Timedelta(hours=xlim[1]) + self.sTime
                        xlim        = (xlim_0,xlim_1)

                stat    = data_da.attrs.get('stat')
                pdct    = pdict.get(stat,{})
                log_z   = pdct.get('log_z',True)

                freqs       = np.sort(data_da['freq_MHz'])[::-1]

                nx      = 100
                ny      = len(freqs)+1

                fig     = plt.figure(figsize=(30,4*ny))

                axs_to_adjust   = []

                inx = 0
                ax  = plt.subplot2grid((ny,nx),(inx,35),colspan=65)
                ax.set_xlim(xlim)
                ax.set_ylim(ylim)

                omni_axs        = geospace_env.omni.plot_dst_kp(self.sTime,self.eTime,ax,xlabels=True,
                                    kp_markersize=10,dst_lw=2,dst_param='SYM-H')
                axs_to_adjust   += omni_axs

                for inx,freq in enumerate(freqs):
                    plt_row = inx+1
                    ax = plt.subplot2grid((ny,nx),(plt_row,0),projection=ccrs.PlateCarree(),colspan=30)

                    ax.coastlines(zorder=10,color='w')
                    ax.plot(np.arange(10))
                    map_data    = map_da.sel(freq_MHz=freq).copy()
                    tf          = map_data < 1
                    map_n       = int(np.sum(map_data))
                    map_data    = np.log10(map_data)
                    map_data.values[tf] = 0
                    map_data.name   = 'log({})'.format(map_data.name)
                    map_data.plot.contourf(x=map_da.attrs['xkey'],y=map_da.attrs['ykey'],ax=ax,levels=30,cmap=mpl.cm.inferno)
                    ax.set_title('')
                    lweight = mpl.rcParams['axes.labelweight']
                    lsize   = mpl.rcParams['axes.labelsize']
                    fdict   = {'weight':lweight,'size':lsize}
                    ax.text(0.5,-0.1,'Radio Spots (N = {!s})'.format(map_n),
                            ha='center',transform=ax.transAxes,fontdict=fdict)
                    
                    ax = plt.subplot2grid((ny,nx),(plt_row,35),colspan=65)
                    data    = data_da.sel(freq_MHz=freq).copy()
                    if log_z:
                        tf          = data < 1.
                        data        = np.log10(data)
                        data.values[tf] = 0
                        data.name   = 'log({})'.format(data.name)

                    robust_dict = self.kwargs.get('robust_dict',{})
                    robust      = robust_dict.get(freq,True)
                    result      = data.plot.contourf(x=data_da.attrs['xkey'],y=data_da.attrs['ykey'],ax=ax,levels=30,robust=robust)

                    for tl in ax.get_xticklabels():
                        tl.set_rotation(10)
                    
                    xlbl    = ax.get_xlabel()
                    if xlbl == 'ut_hrs':
                        ax.set_xlabel('Date Time [UT]')

                    ylbl    = ax.get_ylabel()
                    if ylbl == 'dist_Km':
                        ax.set_ylabel('$R_{gc}$ [km]')

                    ax.set_title('')

                    bdct    = band_dct.get(freq,{})
                    label   = bdct.get('label','{!s} MHz'.format(freq))

                    ax.text(-0.11,0.5,label,transform=ax.transAxes,va='center',
                            rotation=90,fontdict={'weight':'bold','size':30})

                    ax.set_xlim(xlim)
                    ax.set_ylim(ylim)
                    hist_ax = ax

                ########################################
                xpos        = 0.050
                ypos        = 0.950
                fdict       = {'size':32,'weight':'bold'}
                date_str_0  = self.sTime.strftime('%d %b %Y')
                date_str_1  = self.eTime.strftime('%d %b %Y')

                if self.eTime-self.sTime < datetime.timedelta(hours=24):
                    title   = date_str_0
                else:
                    title   = '{!s}-\n{!s}'.format(date_str_0,date_str_1)
                fig.text(xpos,ypos,title,fontdict=fdict)

#                srcs    = '\n'.join([' '+x for x in gl.list_sources(df,bands=meters)])
                srcs    = ''
                txt     = 'Ham Radio Networks\n' + srcs
                fdict   = {'size':30,'weight':'bold'}
                ########################################

                fig.text(xpos,ypos-0.040,txt,fontdict=fdict)

                fig.tight_layout()

                for ax_0 in axs_to_adjust:
                    gl.adjust_axes(ax_0,hist_ax)

                sTime_str   = self.sTime.strftime('%Y%m%d.%H%MUT')
                eTime_str   = self.eTime.strftime('%Y%m%d.%H%MUT')
                date_str    = '-'.join([sTime_str,eTime_str])

                fname   = '.'.join([date_str,self.basename,group,data_var,'png']).replace('.bz2','')
                fpath   = os.path.join(outdir,fname)
                fig.savefig(fpath,bbox_inches='tight')
                print('--> {!s}'.format(fpath))
                plt.close(fig)

def plot_dailies(run_dct):
    sTime   = run_dct['sTime']
    eTime   = run_dct['eTime']
    dates   = daterange(sTime,eTime)

    print('Plotting Dailies: {!s}'.format(run_dct['srcs']))
    for this_sTime in dates:
        this_eTime  = this_sTime + pd.Timedelta('1D')

        rd              = run_dct.copy()
        rd['sTime']     = this_sTime
        rd['eTime']     = this_eTime
        rd['subdir']    = 'dailies'
        nc_obj          = ncLoader(**rd)
        nc_obj.plot(**rd)

def main(run_dct):
    print('Starting main plotting routine...')
    nc_obj      = ncLoader(**run_dct)
    nc_obj.plot(**run_dct)
