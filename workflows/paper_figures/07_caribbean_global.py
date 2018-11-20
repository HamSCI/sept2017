#!/usr/bin/env python3
import os
import datetime

import matplotlib as mpl
mpl.use('Agg')
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

from collections import OrderedDict
from string import ascii_lowercase as letters

from library.timeutils import daterange
from library import gen_lib as gl

from library import goes
from library.omni import Omni
from library import tec_lib

pdct    = OrderedDict()

key = 'Caribbean'
tmp = {}
tmp['file']     = 'data/caribbean_global/caribbean.csv'
tmp['title']    = 'Caribbean RBN & WSPR Spots, 7 & 14 MHz, Rgc <= 5000 km'
pdct[key]   = tmp

key = 'Global'
tmp = {}
tmp['file']     = 'data/caribbean_global/global.csv'
tmp['title']    = 'Global RBN & WSPR Spots, 7 & 14 MHz, Rgc <= 5000 km'
pdct[key]   = tmp

tmp = {}
#tmp['kp_markersize']    = 20
tmp['title_size']       = 16
tmp['ticklabel_size']   = 12
tmp['label_size']       = 14
tmp['legend_size']      = 12
tmp['kp_markersize']    = 5
lout    = tmp

def time_range(sTime,eTime,step=datetime.timedelta(hours=24)):
    times   = [sTime]
    while times[-1] < eTime:
        times.append(times[-1]+step)
    return times

def format_timeticklabels(ax):
    xtls    = []
    for inx,xtk in enumerate(ax.get_xticks()):
        if inx == 0:
            fmt = '%d %b %Y'
        else:
            fmt = '%d %b'

        xtkd    = mpl.dates.num2date(xtk)
        xtls.append(xtkd.strftime(fmt))
    ax.set_xticklabels(xtls)

def set_text_props(title_size='xx-large',ticklabel_size='xx-large',
        label_size='xx-large',legend_size='large',text_weight='bold',**kwargs):
    rcp = mpl.rcParams
    rcp['figure.titlesize']     = title_size 
    rcp['axes.titlesize']       = title_size 
    rcp['axes.labelsize']       = label_size
    rcp['xtick.labelsize']      = ticklabel_size 
    rcp['ytick.labelsize']      = ticklabel_size 
    rcp['legend.fontsize']      = legend_size

    rcp['figure.titleweight']   = text_weight
    rcp['axes.titleweight']     = text_weight
    rcp['axes.labelweight']     = text_weight

set_text_props(**lout)

class StackPlot(object):
    def __init__(self):
        output_dir  = 'output/galleries/caribbean_global'
        gl.prep_output({0:output_dir},clear=False)

        self.sTime  = datetime.datetime(2017,9,4)
        self.eTime  = datetime.datetime(2017,9,30)

        self.nrows  = 3 + len(pdct)
        self.ncols  = 1
        self.plt_nr = 0

        self.fig        = plt.figure(figsize=(18,12))

        self.ts_axs     = []

        self.kpDst()
        self.goes()
        self.hamRadio()
#        self.tec(region='Greater Caribbean')
        self.tec(region='US')

        for ax in self.ts_axs:
            format_timeticklabels(ax)
    
        # Mark start of geomagnetic activity. ## 
        mark_times  = []
        mark_times.append({'val':datetime.datetime(2017,9,7,21),'label':'2100'})
        for ax_inx, ax in enumerate(self.ts_axs):
            color   = 'k'
            lw      = 2
            ls      = ':'
            for mtd in mark_times:
                mark_time   = mtd.get('val')
                label       = mtd.get('label')

                ax.axvline(mark_time,color=color,lw=lw,ls=ls)
                if ax_inx == 0:
                    trans   = mpl.transforms.blended_transform_factory(ax.transData, ax.transAxes)
                    ax.text(mark_time,1,label,rotation=90,va='top',ha='right',transform=trans,fontdict={'weight':'bold'},color=color)
        ########################################

        ax  = self.fig.gca()
        ax.set_xlabel('Date Time [UT]')

        self.fig.tight_layout()
        fname   = 'caribbean_global.png'
        fpath   = os.path.join(output_dir,fname)
        self.fig.savefig(fpath,bbox_inches='tight')
        print(fpath)

    def next_ax(self):
        self.plt_nr     += 1
        ax              = self.fig.add_subplot(self.nrows,self.ncols,self.plt_nr)

        xltr            = -0.075
        yltr            = 0.95
        fltr            = {'weight':'bold','size':24}
        ltr = '({!s})'.format(letters[self.plt_nr-1])
        ax.text(xltr,yltr,ltr,transform=ax.transAxes,fontdict=fltr)
        return ax

    def kpDst(self):
        ax  = self.next_ax()

        omni            = Omni()
        msize           = lout.get('kp_markersize',10)
        dst_lw          = lout.get('goes_lw',2)
        omni_axs        = omni.plot_dst_kp(self.sTime,self.eTime,ax,xlabels=True,
                            kp_markersize=msize,dst_lw=dst_lw,dst_param='SYM-H')

        for ax in omni_axs:
            self.ts_axs.append(ax)

    def goes(self):
        ax  = self.next_ax()

        goes_dcts       = OrderedDict()
        goes_dcts[13]   = {}
        goes_dcts[15]   = {}

        flares_combined = pd.DataFrame()
        for sat_nr,gd in goes_dcts.items():
            gd['data']      = goes.read_goes(self.sTime,self.eTime,sat_nr=sat_nr)
            flares          = goes.find_flares(gd['data'],min_class='M1',window_minutes=60)
            flares['sat']   = sat_nr
            gd['flares']    = flares
            flares_combined = flares_combined.append(flares).sort_index()
            gd['var_tags']  = ['B_AVG']
            gd['labels']    = ['GOES {!s}'.format(sat_nr)]

        flares_combined     = flares_combined[~flares_combined.index.duplicated()].sort_index()

        xkey            = 'occurred'
        xdct            = gl.prmd[xkey]
        xlabel          = xdct.get('label',xkey)
        goes_lw         = lout.get('goes_lw',2)
        for sat_nr,gd in goes_dcts.items():
            goes.goes_plot(gd['data'],self.sTime,self.eTime,ax=ax,
                    var_tags=gd['var_tags'],labels=gd['labels'],
                    legendLoc='upper right',lw=goes_lw)

        title   = 'NOAA GOES X-Ray (0.1 - 0.8 nm) Irradiance'
        size    = lout.get('label_size',20)
        ax.text(0.01,0.05,title,transform=ax.transAxes,ha='left',fontdict={'size':size,'weight':'bold'})
        self.ts_axs.append(ax)

    def hamRadio(self):
        for inx,(key,dct) in enumerate(pdct.items()):
            print(key)
            print(dct['file'])

            df  = pd.read_csv(dct['file'],parse_dates=[0],names=['datetime','spots'],header=0)

            dates   = list(daterange(self.sTime, self.eTime))
            vals    = []
            for date in dates:
                tf  = np.logical_and(df['datetime'] >= date,
                                     df['datetime'] <  date+datetime.timedelta(days=1))
                val = df[tf].mean()
                vals.append(val)
            vals    = np.array(vals)

            ax  = self.next_ax()

    #        xx  = df['datetime']
    #        yy  = df['spots']

            xx  = dates
            yy  = vals
            goes_lw         = lout.get('goes_lw',2)
            ax.plot(xx,yy,label=None,lw=goes_lw,marker='o')

            ax.axhline(np.nanmean(yy),ls='--',label='mean')
            ax.set_title(dct.get('title',key))

            ax.set_ylabel('Daily Spot\nAverage')
            ax.set_xlim(self.sTime,self.eTime)
            ax.grid()
            self.ts_axs.append(ax)

    def tec(self,region='Greater Caribbean',param='tec',data_dir='data/tec',download=False):
        sTime   = self.sTime
        eTime   = self.eTime
        if download:
            tec_lib.download_tec(self.sTime,self.eTime,data_dir=data_dir)

#        t0_str  = self.sTime.strftime('%Y%m%d.%H%M')
#        t1_str  = self.eTime.strftime('%Y%m%d.%H%M')
#        csvname = '{!s}-{!s}.cache.csv.bz2'.format(t0_str,t1_str)
#        csvpath = os.path.join(data_dir,csvname)
#        if not os.path.exists(csvpath):
#            print('Creating {!s}'.format(csvpath))
#            df  = tec_lib.load_tec(self.sTime,self.eTime,data_dir)
#            df.to_csv(csvpath,index=False,compression='bz2')
#        else:
#            print('Loading {!s}'.format(csvpath))
#            df  = pd.read_csv(csvpath,parse_dates=['datetime'])

        df  = tec_lib.load_tec(self.sTime,self.eTime,data_dir,region=region)

        step    = datetime.timedelta(hours=24)
        win     = step
        dates   = time_range(self.sTime, self.eTime,step)

        if win < datetime.timedelta(hours=1):
            win_label   = '{:.0f} min Bins'.format(win.total_seconds()/60.)
        elif win < datetime.timedelta(days=1):
            win_label   = '{:.0f} hr Bins'.format(win.total_seconds()/3600.)
        else:
            win_label   = '{:.0f} day Bins'.format(win.total_seconds()/86400.)

        vals    = []
        for date in dates:
            tf  = np.logical_and(df['datetime'] >= date,
                                 df['datetime'] <  date+win)
            val = df[param][tf].mean()
            vals.append(val)
        vals    = np.array(vals)

        ax  = self.next_ax()

#        xx  = df['datetime']
#        yy  = df['spots']

        xx  = dates
        yy  = vals
        goes_lw         = lout.get('goes_lw',2)
        ax.plot(xx,yy,label=None,lw=goes_lw,marker='o')
        ax.axhline(np.nanmean(yy),ls='--',label='mean')
        title = 'TEC - {!s} ({!s})'.format(region,win_label)
        ax.set_title(title)

        ax.set_ylabel('Mean TEC [TECU]')
        ax.set_xlim(self.sTime,self.eTime)
        ax.grid()
        self.ts_axs.append(ax)

if __name__ == '__main__':
    StackPlot()

