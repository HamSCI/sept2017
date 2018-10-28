#!/usr/bin/env python3
"""
Script covering the entire histogram workflow process.
"""
import os
import datetime
import library as lib

data_dir_0  = 'data/histograms/0-10000km_dx30min_dy500km'

run_name    = 'test_filter'
data_dir    = os.path.join('data/histograms',run_name)
plot_dir    = os.path.join('output/galleries/histograms',run_name)
params      = ['spot_density']
xkeys       = ['ut_hrs','slt_mid']
sTime       = datetime.datetime(2017,9,1)
eTime       = datetime.datetime(2017,9,3)

geo_env     = lib.GeospaceEnv()

# Create histogram NetCDF Files ################################################
rd  = {}
rd['input_dir']             = data_dir_0
rd['output_dir']            = data_dir
rd['geospace_env']          = geo_env
rd['symh_min']              = -25
rd['symh_max']              =  25
rd['kp_min']                = None
rd['kp_max']                = 3
rd['goes_min']              = -25
rd['goes_max']              =  25
lib.select_histograms.main(rd)
