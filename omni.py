import datetime
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
from matplotlib import pyplot as plt

def to_ut_hr(dt):
    return dt.hour + dt.minute/60. + dt.second/3600.

class Omni():
    def __init__(self):
        # Load OMNI ############################
        omni_csv        = 'data/omni/omni2_3051.csv'
        omni_df         = pd.read_csv(omni_csv,
                            parse_dates={'datetime':['year','doy','hr']},
                            date_parser=self._date_parser)
        omni_df         = omni_df.set_index('datetime')
        omni_df['Kp']   = omni_df['Kp_x10']/10.
        del omni_df['Kp_x10']
        self.df     = omni_df

        # Load Kyoto SYM-H #####################
        fpath           ='data/kyoto_wdc/WWW_aeasy00025746.dat.txt'
        df_symasy       = pd.read_csv(fpath,sep='\s+',header=14,parse_dates=[['DATE','TIME']])
        df_symasy       = df_symasy.set_index('DATE_TIME')
        df_symasy       = df_symasy[['ASY-D','ASY-H','SYM-D','SYM-H']].copy()
        self.df_symasy  = df_symasy

    def _date_parser(self,years,doys,hrs):
        datetimes = []
        for year,doy,hr in zip(years,doys,hrs):
            dt  = datetime.datetime(int(year),1,1)+datetime.timedelta(days=(int(doy)-1),hours=int(hr))
            datetimes.append(dt)
        return datetimes

    def plot_dst_kp(self,sTime,eTime,ax,xkey='index',xlabels=True,
            dst_param='Dst_nT',dst_lw=1,kp_markersize=10):
        """
        DST and Kp

        dst_param = ['Dst_nT','SYM-H']
        """
        tf  = np.logical_and(self.df.index >= sTime, self.df.index < eTime)
        df  = self.df[tf].copy()

        ut_hrs  = df.index.map(to_ut_hr)

        lines       =[]

        if dst_param == 'Dst_nT':
            if xkey == 'index':
                xx      = df.index
                xlim    = (sTime,eTime)
            else:
                xx      = ut_hrs
                xlim    = (to_ut_hr(sTime), (eTime-sTime).total_seconds()/3600.)

            yy      = df['Dst_nT'].tolist()
            ylabel  = 'Dst [nT]'
        else:
            xx      = self.df_symasy.index
            yy      = self.df_symasy[dst_param].tolist()
            xlim    = (sTime,eTime)
            ylabel  = 'SYM-H [nT]'

        tmp,        = ax.plot(xx,yy,label=ylabel,color='k',lw=dst_lw)
#        ax.fill_between(xx,0,yy,color='0.75')
        lines.append(tmp)
        ax.set_ylabel(ylabel)
        ax.axhline(0,color='k',ls='--')
        ax.set_ylim(-200,100)
        ax.set_xlim(xlim)

        # Kp ###################################
        ax_1        = ax.twinx()
        ax.set_zorder(ax_1.get_zorder()+1)
        ax.patch.set_visible(False)
        low_color   = 'green'
        mid_color   = 'darkorange'
        high_color  = 'red'
        label       = 'Kp'

        if xkey == 'index':
            xvals       = df.index
        else:
            xvals       = np.array(ut_hrs)

        kp          = np.array(df['Kp'].tolist())

        if len(kp) > 0:
            color       = low_color
            markers,stems,base  = ax_1.stem(xvals,kp)
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
                markers,stems,base  = ax_1.stem(xx,yy)
                for stem in stems:
                    stem.set_color(color)
                markers.set_color(color)
                markers.set_markersize(kp_markersize)
                lines.append(markers)

            tf = kp >= 5
            if np.count_nonzero(tf) > 0:
                xx      = xvals[tf]
                yy      = kp[tf]
                color   = high_color
                markers,stems,base  = ax_1.stem(xx,yy)
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

        if not xlabels:
            for xtl in ax.get_xticklabels():
                xtl.set_visible(False)
        plt.sca(ax)
        return [ax,ax_1]
