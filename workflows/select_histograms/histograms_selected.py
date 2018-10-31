#!/usr/bin/env python3
"""
Script covering the entire histogram workflow process.
"""
import os
import datetime
import library as lib

data_dir_0  = 'data/histograms/0-10000km_dx30min_dy500km'

run_name    = 'global_quiet_baseline'
data_dir    = os.path.join('data/histograms',run_name)
plot_dir    = os.path.join('output/galleries/histograms',run_name)
params      = ['spot_density']
xkeys       = ['ut_hrs','slt_mid']
sTime       = datetime.datetime(2016,1,1)
eTime       = datetime.datetime(2018,1,1)

geo_env     = lib.GeospaceEnv()

# Create histogram NetCDF Files ################################################
rd  = {}
rd['sDate']                 = sTime
rd['eDate']                 = eTime
rd['params']                = params
rd['xkeys']                 = xkeys
rd['rgc_lim']               = (0,10000)
rd['filter_region']         = None
rd['filter_region_kind']    = 'mids'
rd['xb_size_min']           = 30.
rd['yb_size_km']            = 500.
rd['output_dir']            = data_dir_0
rd['band_obj']              = lib.gl.BandData()
#lib.calculate_histograms.main(rd)

# Create histogram NetCDF Files ################################################
rd  = {}
rd['input_dir']             = data_dir_0
rd['output_dir']            = data_dir
rd['geospace_env']          = geo_env
rd['symh_min']              = -25
rd['symh_max']              =  25
rd['kp_min']                = None
rd['kp_max']                = 3
lib.select_histograms.main(rd)

# Calculate Statistics from Histograms #########################################
rd = {}
rd['src_dir']               = data_dir
rd['params']                = params
rd['xkeys']                 = xkeys
rd['stats']                 = ['sum','mean','median','std']
lib.statistics_histograms.main(rd)

# Baseline daily observations against statistics ###############################
rd = {}
rd['src_dir']               = data_dir
rd['xkeys']                 = xkeys
rd['stats']                 = ['pct_err','z_score']
lib.baseline_histograms.main(rd)

# Visualization ################################################################
### Visualize Observations
rd = {}
rd['srcs']                  = os.path.join(data_dir,'*.data.nc')
rd['baseout_dir']           = plot_dir
rd['sTime']                 = sTime
rd['eTime']                 = eTime
rd['geospace_env']          = geo_env
lib.visualize_histograms.main(rd)
lib.visualize_histograms.plot_dailies(rd)

### Visualize Baselines
rd['srcs']                  = os.path.join(data_dir,'*.baseline_compare.nc')
lib.visualize_histograms.main(rd)
lib.visualize_histograms.plot_dailies(rd)

### Visualize Statistics
rd = {}
rd['srcs']                  = os.path.join(data_dir,'stats.nc')
rd['baseout_dir']           = plot_dir
lib.visualize_histograms_simple.main(rd)
import ipdb; ipdb.set_trace()
