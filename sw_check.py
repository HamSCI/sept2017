#!/usr/bin/python3
import os
from collections import OrderedDict
import datetime

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

import numpy as np

from omni import Omni
import goes
import gen_lib as gl

def plot_sw(sTime,eTime,xkey='occurred',output_dir='output',lout={}):

    date_str_0  = sTime.strftime('%Y%m%d.%H%M')
    date_str_1  = eTime.strftime('%Y%m%d.%H%M')
    date_str    = '{!s}-{!s}'.format(date_str_0,date_str_1)

    fig = plt.figure(figsize=(20,8))

    nrow    = 2
    ncol    = 1
    nplt    = 0

    nplt    += 1
    ax      = fig.add_subplot(nrow,ncol,nplt)

    # Dst and Kp ###########################
    omni            = Omni()
    msize           = lout.get('kp_markersize',10)
    dst_lw          = lout.get('goes_lw',2)
    omni_axs        = omni.plot_dst_kp(sTime,eTime,ax,xlabels=True,
                        kp_markersize=msize,dst_lw=dst_lw)


    ax.grid(axis='x')
    ax.grid(axis='x',which='minor')

    tf  = np.logical_and(omni.df.index >= sTime,
                         omni.df.index <  eTime)
    odf = omni.df[tf].copy()

    # GOES X-Ray ###########################
    nplt    += 1
    ax      = fig.add_subplot(nrow,ncol,nplt)
    
    goes_dcts       = OrderedDict()
    goes_dcts[13]   = {'marker':'*','markersize':15,'color':'blue'}
    goes_dcts[15]   = {'marker':'o','markersize': 5,'color':'orange'}

    for sat_nr,gd in goes_dcts.items():
        gd['data']      = goes.read_goes(sTime,eTime,sat_nr=sat_nr)
        gd['flares']    = goes.find_flares(gd['data'],min_class='M1',window_minutes=60)
        gd['var_tags']  = ['B_AVG']
        gd['labels']    = ['GOES {!s}'.format(sat_nr)]

    xdct            = gl.prmd[xkey]
    xlabel          = xdct.get('label',xkey)
    goes_lw         = lout.get('goes_lw',2)
    for sat_nr,gd in goes_dcts.items():
        goes.goes_plot(gd['data'],sTime,eTime,ax=ax,
                var_tags=gd['var_tags'],labels=gd['labels'],
                legendLoc='upper right',lw=goes_lw)

        flares  = gd['flares']
        with open(os.path.join(output_dir,'{!s}-G{!s}-flares.txt'.format(date_str,sat_nr)),'w') as fl:
            fl.write(flares.to_string())

        for key,flare in flares.iterrows():
            flr_plt_dct = {}
#            flr_plt_dct['label']        = '{0} Class Flare @ {1}'.format(flare['class'],key.strftime('%H%M UT'))
            flr_plt_dct['color']        = gd.get('color','blue')
            flr_plt_dct['marker']       = gd.get('marker','o')
            flr_plt_dct['markersize']   = gd.get('markersize',10)
            ax.plot(key,flare['B_AVG'],**flr_plt_dct)
    ########################################

    title   = 'NOAA GOES X-Ray (0.1 - 0.8 nm) Irradiance'
    size    = lout.get('label_size',20)
    ax.text(0.01,0.05,title,transform=ax.transAxes,ha='left',fontdict={'size':size,'weight':'bold'})

    fname   = 'sw-{!s}.png'.format(date_str)
    fpath   = os.path.join(output_dir,fname)
    fig.savefig(fpath,bbox_inches='tight')
    plt.close(fig)

if __name__ == '__main__':
    output_dir  = 'output/galleries/sw_check'
    gl.prep_output({0:output_dir},clear=True,php=False)

    run_dcts = []
    rd = {}
    rd['sTime']         = datetime.datetime(2017, 9,  7)
    rd['eTime']         = datetime.datetime(2017, 9, 10)
    rd['output_dir']    = output_dir
    run_dcts.append(rd)

    rd = {}
    rd['sTime']         = datetime.datetime(2017, 9,  7,18)
    rd['eTime']         = datetime.datetime(2017, 9,  9, 6)
    rd['output_dir']    = output_dir
    run_dcts.append(rd)

    rd = {}
    rd['sTime']         = datetime.datetime(2017, 9,  7,18)
    rd['eTime']         = datetime.datetime(2017, 9,  8, 0)
    rd['output_dir']    = output_dir
    run_dcts.append(rd)

    rd = {}
    rd['sTime']         = datetime.datetime(2017, 9,  8,21)
    rd['eTime']         = datetime.datetime(2017, 9,  9, 3)
    rd['output_dir']    = output_dir
    run_dcts.append(rd)
    
    for rd in run_dcts:
        plot_sw(**rd)
