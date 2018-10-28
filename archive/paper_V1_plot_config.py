
#    dct = {}
#    dct['sTime']                = datetime.datetime(2017,  9, 1)
#    dct['eTime']                = datetime.datetime(2017, 10, 1)
##    dct['rgc_lim']              = (0,20000)
#    dct['maplim_region']        = 'World'
#    dct['log_hist']             = True
#    dct['output_dir']           = output_dir
#    dct['fname']                = 'SEPT2017_WSPR_RBN'
#    dct['generate_csv']         = True
#    run_dcts.append(dct)

#    dct = {}
#    dct['sTime']                = datetime.datetime(2017, 9, 6,6)
#    dct['eTime']                = datetime.datetime(2017, 9, 6,18)
#    dct['rgc_lim']              = (0,3000)
#    dct['maplim_region']        = 'Europe'
#    dct['filter_region']        =  dct['maplim_region']
#    dct['solar_zenith_region']  =  dct['maplim_region']
#    dct['filter_region_kind']   = 'mids'
#    dct['band_obj']             = gl.BandData([7,14,21,28])
#    dct['layout']               = '4band12hr'
#    dct['output_dir']           = output_dir
#    dct['fname']                = 'flareEU'
#    run_dcts.append(dct)

    dct = {}
    dct['sTime']                = datetime.datetime(2017, 9, 6,6)
    dct['eTime']                = datetime.datetime(2017, 9, 6,18)
    dct['rgc_lim']              = (0,3000)
    dct['maplim_region']        = 'US'
    dct['filter_region']        =  dct['maplim_region']
    dct['solar_zenith_region']  =  dct['maplim_region']
    dct['filter_region_kind']   = 'mids'
    dct['band_obj']             = gl.BandData([7,14,21,28])
    dct['layout']               = '4band12hr'
    dct['output_dir']           = output_dir
    dct['fname']                = 'flareUS'
    run_dcts.append(dct)

#    dct = {}
#    dct['sTime']                = datetime.datetime(2017, 9,  7)
#    dct['eTime']                = datetime.datetime(2017, 9, 10)
#    dct['rgc_lim']              = (0,10000)
#    dct['maplim_region']        = 'World'
#    dct['filter_region']        =  dct['maplim_region']
#    dct['filter_region_kind']   = 'mids'
#    dct['layout']               = '6band3day'
#    dct['log_hist']             = True
#    dct['output_dir']           = output_dir
#    dct['fname']                = 'geomag'
#    run_dcts.append(dct)
#
#    dct = {}
#    dct['sTime']                = datetime.datetime(2017, 9, 4)
#    dct['eTime']                = datetime.datetime(2017, 9, 14)
#    dct['rgc_lim']              = (0,5000)
#    dct['maplim_region']        = 'Greater Greater Caribbean'
#    dct['box']                  = 'Greater Caribbean'
#    dct['solar_zenith_region']  = dct['box']
#    dct['filter_region']        = dct['box']
#    dct['filter_region_kind']   = 'endpoints'
#    dct['log_hist']             = True
#    dct['band_obj']             = gl.BandData([7,14])
#    dct['map_midpoints']        = False
#    dct['map_filter_region']    = True
#    dct['layout']               = '2band'
#    dct['output_dir']           = output_dir
#    dct['fname']                = 'caribbean'
##    dct['find_flares']          = True
#    dct['flare_labels']         = False
#    run_dcts.append(dct)
#
##    dct = dct.copy()
##    del dct['band_obj']
##    del dct['layout']
##    dct['fname']                = 'caribbean-6band'
##    run_dcts.append(dct)
#
#    dct = {}
#    dct['sTime']                = datetime.datetime(2017, 9, 1)
#    dct['eTime']                = datetime.datetime(2017, 10, 1)
#    dct['rgc_lim']              = (0,20000)
#    dct['maplim_region']        = 'World'
#    dct['log_hist']             = True
#    dct['output_dir']           = output_dir
#    dct['fname']                = 'summary'
#    run_dcts.append(dct)

