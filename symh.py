#!/usr/bin/env python3
import os
import datetime
import glob

import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
from matplotlib import pyplot as plt

from collections import OrderedDict

import gen_lib as gl

this_name   = os.path.basename(__file__[:-3])
output_dir  = os.path.join('output',this_name)
gl.prep_output({0:output_dir})

def get_max(row,window,series):
    sTime   = row[0]
    td      = pd.Timedelta(window)

    tf      = np.logical_and(series.index >= sTime, series.index < sTime+td)
    tmp     = series[tf]
    idxmax  = series.idxmax()
    mx      = series.max()
    return (idxmax,mx)


class SymH():
    def __init__(self,years=[2017],pattern='data/kyoto_wdc/SYM-{!s}.dat.txt',output_dir='output'):
        self.output_dir = output_dir
        
        # Load Kyoto SYM-H #####################
        if years is None:
            fpaths          = glob.glob(pattern.format('*'))
        else:
            fpaths          = [pattern.format(x) for x in years]

        df = pd.DataFrame()
        for fpath in fpaths:
            df_tmp      = pd.read_csv(fpath,sep='\s+',header=14,parse_dates=[['DATE','TIME']])
            df_tmp      = df_tmp.set_index('DATE_TIME')
            df_tmp      = df_tmp[['ASY-D','ASY-H','SYM-D','SYM-H']].copy()
            df          = df.append(df_tmp)

        df.sort_index(inplace=True)
        tf      = df[:] == 99999
        df[tf]  = np.nan
        df.dropna(inplace=True)
        self.df = df
    def plot(self,var='SYM-H',figsize=(10,8)):
        fig = plt.figure(figsize=figsize)
        ax  = fig.add_subplot(1,1,1)
        xx  = self.df.index
        yy  = self.df[var]
        ax.plot(xx,yy)
        ax.set_xlabel('UT Time')
        ax.set_ylabel(var)
        fig.tight_layout()
        fname   = '{!s}.png'.format(var)
        fpath   = os.path.join(self.output_dir,fname)
        fig.savefig(fpath,bbox_inches='tight')

    def detect_storms(self,window='6H',ssc_min=0,mainPhase_max=-50):
        t0  = datetime.datetime.now()
        print('Storm Detection')

        # Calculate the max and min SYM-H value in each rolling window.
        df          = self.df['SYM-H'].to_frame()
        df_min      = df.rolling(window).min()
        df_max      = df.rolling(window).max()

        df['min']   = df_min
        df['max']   = df_max
        
        # Eliminate any windows where the max and min are outside of the
        # specified thresholds.
        tf          = np.logical_and(df['max'] > ssc_min, df['min'] <= mainPhase_max)
        df_stormFind    = df[tf].copy()
        df_stormFind.sort_index(inplace=True)

        # Calculate the slope on the max values. The max for each storm period should
        # monotonically decrease. Then, remove all zero/negative values.
        smaxes      = df_stormFind['max']
        dsmaxes     = smaxes.diff()
        tf          = dsmaxes <= 0
        dsmaxes[tf] = np.nan
        dsmaxes.dropna(inplace=True)

        # Create a column that only has the SYM-H max values at their occurrence time
        # and NaN everywhere else.
        smax                    = df_stormFind.loc[dsmaxes.index,'max'] 
        df_stormFind['smax']    = smax
        self.df_stormFind       = df_stormFind

        df_storm                = smax.to_frame()
        attrs                   = OrderedDict()
        attrs['window']         = window
        attrs['ssc_min']        = ssc_min
        attrs['mainPhase_max']  = mainPhase_max
        self.df_storm           = df_storm
        self.detect_attrs       = attrs

        # Plot the Storm Finder results as a test.
        self.plot_df_stormFind()

        # Generate the Storm Report
        fname   = 'storm_report.csv'
        fpath   = os.path.join(self.output_dir,fname)
        with open(fpath,'w') as fl:
            fl.write('# Geomagnetic Storm Detection Report\n')
            for key,val in attrs.items():
                line    = '# {!s}: {!s}\n'.format(key,val)
                fl.write(line)

        df_storm.to_csv(fpath,mode='a')

        t1  = datetime.datetime.now()
        print(' --> {!s} s'.format((t1-t0).total_seconds()))
        import ipdb; ipdb.set_trace()

    def plot_df_stormFind(self):
        df_stormFind    = self.df_stormFind
        fig = plt.figure(figsize=(15,8))
        ax  = fig.add_subplot(1,1,1)

        var = 'max'
        xx  = df_stormFind.index
        yy  = df_stormFind[var]
        ax.plot(xx,yy,label=var,marker='*')

        var = 'smax'
        xx  = df_stormFind.index
        yy  = df_stormFind[var]
        ax.plot(xx,yy,label=var,marker='o')

        ax.set_xlabel('UT Time')
        ax.set_ylabel(var)
        ax.legend()
        fig.tight_layout()
        fname   = '{!s}.png'.format(var)
        fpath   = os.path.join(self.output_dir,fname)
        fig.savefig(fpath,bbox_inches='tight')


def main():
    symh = SymH(output_dir=output_dir)
    symh.detect_storms()
#    symh.plot()


if __name__ == '__main__':
    main()
