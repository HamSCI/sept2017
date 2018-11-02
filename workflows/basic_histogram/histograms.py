#!/usr/bin/env python3
"""
Script covering the entire histogram workflow process.
"""
import os
import datetime
import library as lib

run_name    = 'testing'
data_dir    = os.path.join('data/histograms',run_name)
plot_dir    = os.path.join('output/galleries/histograms',run_name)
params      = ['spot_density']
xkeys       = ['ut_hrs','slt_mid']
sTime       = datetime.datetime(2017,9,6,6)
eTime       = datetime.datetime(2017,9,6,18)
region      = 'US'
rgc_lim     = (0, 3000)

geo_env     = lib.GeospaceEnv()

## Create histogram NetCDF Files ################################################
#rd  = {}
#rd['sDate']                 = sTime
#rd['eDate']                 = eTime
#rd['params']                = params
#rd['xkeys']                 = xkeys
#rd['rgc_lim']               = rgc_lim
#rd['filter_region']         = region
#rd['filter_region_kind']    = 'mids'
#rd['xb_size_min']           = 30.
#rd['yb_size_km']            = 500.
#rd['reprocess']             = True
#rd['output_dir']            = data_dir
#rd['band_obj']              = lib.gl.BandData()
#lib.calculate_histograms.main(rd)
#
## Calculate Statistics from Histograms #########################################
#rd = {}
#rd['src_dir']               = data_dir
#rd['params']                = params
#rd['xkeys']                 = xkeys
#rd['stats']                 = ['sum','mean','median','std']
#lib.statistics_histograms.main(rd)
#
## Baseline daily observations against statistics ###############################
#rd = {}
#rd['src_dir']               = data_dir
#rd['xkeys']                 = xkeys
#rd['stats']                 = ['pct_err','z_score']
#lib.baseline_histograms.main(rd)

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
lib.visualize_histograms.main(rd)
import ipdb; ipdb.set_trace()
lib.visualize_histograms.plot_dailies(rd)

### Visualize Baselines
rd['srcs']                  = os.path.join(data_dir,'*.baseline_compare.nc.bz2')
lib.visualize_histograms.main(rd)
lib.visualize_histograms.plot_dailies(rd)

### Visualize Statistics
rd = {}
rd['srcs']                  = os.path.join(data_dir,'stats.nc.bz2')
rd['baseout_dir']           = plot_dir
lib.visualize_histograms_simple.main(rd)
import ipdb; ipdb.set_trace()
