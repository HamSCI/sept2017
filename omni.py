import datetime
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
from matplotlib import pyplot as plt

class Omni():
    def __init__(self):
        omni_csv        = 'data/omni/omni2_3051.csv'
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

        if not xlabels:
            for xtl in ax.get_xticklabels():
                xtl.set_visible(False)
        plt.sca(ax)
        return [ax,ax_1]
