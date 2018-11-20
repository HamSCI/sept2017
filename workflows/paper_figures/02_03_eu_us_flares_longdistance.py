#!/usr/bin/env python3
"""
Script covering the entire histogram workflow process.
"""
import os
import datetime
import library as lib

import matplotlib as mpl
mpl.use('Agg')

mpl.rcParams['font.size'] = 13

#'font.size': 10.0
#'xtick.labelsize': 'xx-large'
#'ytick.labelsize': 'xx-large'
#'axes.labelsize': 'xx-large'
#'axes.titlesize': 'xx-large'
#'figure.titlesize': 'xx-large'
#'legend.fontsize': 'large'

def main(run_name):
    data_dir    = os.path.join('data/histograms',run_name+'long_dist')
    plot_dir    = os.path.join('output/galleries/histograms',run_name+'long_dist')
    params      = ['spot_density']
    xkeys       = ['ut_hrs']
    sTime       = datetime.datetime(2017,9,6,6)
    eTime       = datetime.datetime(2017,9,6,18)
    region      = run_name
    rgc_lim     = (0, 40000)

    geo_env     = lib.GeospaceEnv()

    # Create histogram NetCDF Files ################################################
    rd  = {}
    rd['sDate']                 = sTime
    rd['eDate']                 = eTime
    rd['params']                = params
    rd['xkeys']                 = xkeys
    rd['rgc_lim']               = rgc_lim
    rd['filter_region']         = region
    rd['filter_region_kind']    = 'mids'
    rd['xb_size_min']           = 10.
    rd['yb_size_km']            = 250.
    rd['reprocess']             = True
    rd['output_dir']            = data_dir
    rd['band_obj']              = lib.gl.BandData()
    lib.calculate_histograms.main(rd)

    # Visualization ################################################################
    ### Visualize Observations
    rd = {}
    rd['srcs']                  = os.path.join(data_dir,'*.data.nc.bz2')
    rd['baseout_dir']           = plot_dir
    rd['sTime']                 = sTime
    rd['eTime']                 = eTime
    rd['plot_region']           = region
    rd['geospace_env']          = geo_env
    rd['band_keys']             = [28, 21, 14, 7]
    axv = []
    axv.append(datetime.datetime(2017,9,6, 8,57))
    axv.append(datetime.datetime(2017,9,6,11,53))
    rd['axvlines']              = axv
    lib.visualize_histograms.main(rd)

if __name__ == '__main__':
    run_names = []
    run_names.append('Europe')
    run_names.append('US')

    for rn in run_names:
        main(rn)
