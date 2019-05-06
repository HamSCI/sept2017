#!/usr/bin/env python3
"""
Script covering the entire histogram workflow process.
"""
import os
import datetime
import library as lib

#run_name    = 'Europe'
#run_name    = 'World'
run_name    = 'US'
data_dir    = os.path.join('data/wave_search',run_name)
plot_dir    = os.path.join('output/galleries/wave_search',run_name)
params      = ['spot_density']
xkeys       = ['ut_hrs']
#sTime       = datetime.datetime(2017,11,1)
#eTime       = datetime.datetime(2017,12,31)
sTime       = datetime.datetime(2017,11,3)
eTime       = datetime.datetime(2017,11,3)
region      = run_name
rgc_lim     = (0.,3000)

geo_env     = lib.GeospaceEnv()

# Create histogram NetCDF Files ################################################
rd  = {}
rd['sDate']                 = sTime
rd['eDate']                 = eTime
rd['params']                = params
rd['xkeys']                 = xkeys
rd['rgc_lim']               = rgc_lim
rd['filter_region']         = run_name
rd['filter_region_kind']    = 'mids'
rd['xb_size_min']           = 2.
rd['yb_size_km']            = 25.
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
rd['band_keys']             = [14, 7]

#lib.visualize_histograms.main(rd)
lib.visualize_histograms.plot_dailies(rd)
import ipdb; ipdb.set_trace()
