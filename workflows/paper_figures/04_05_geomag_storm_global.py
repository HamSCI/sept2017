#!/usr/bin/env python3

"""
Script covering the entire histogram workflow process.
"""
import os
import shutil
import glob
import datetime
import library as lib

import matplotlib as mpl
mpl.use('Agg')

mpl.rcParams['font.size'] = 13

geo_env     = lib.GeospaceEnv()

class HamDataSet(object):
    def __init__(self,run_dct={}):
        run_dct['params']               = run_dct.get('params',['spot_density'])
        run_dct['xkeys']                = run_dct.get('xkeys', ['ut_hrs'])

        run_dct['xb_size_min']          = run_dct.get('xb_size_min',  30.)
        run_dct['yb_size_km']           = run_dct.get('yb_size_km',  500.)
        run_dct['filter_region_kind']   = run_dct.get('filter_region_kind', 'mids')
        self.rd = run_dct

    def create(self):
        # Create histogram NetCDF Files ################################################
        rd  = {}
        rd['sDate']                 = self.rd.get('sTime')
        rd['eDate']                 = self.rd.get('eTime')
        rd['params']                = self.rd.get('params')
        rd['xkeys']                 = self.rd.get('xkeys')
        rd['rgc_lim']               = self.rd.get('rgc_lim')
        rd['filter_region']         = self.rd.get('filter_region')
        rd['filter_region_kind']    = self.rd.get('filter_region_kind')
        rd['xb_size_min']           = self.rd.get('xb_size_min')
        rd['yb_size_km']            = self.rd.get('yb_size_km')
        rd['base_dir']              = self.rd.get('base_dir')
        rd['reprocess']             = self.rd.get('reprocess')
        rd['band_obj']              = lib.gl.BandData()
        data_dir_0                  = lib.calculate_histograms.main(rd)

        self.rd['data_dir_0']       = data_dir_0

    def select(self):
        # Create histogram NetCDF Files ################################################
        rd  = {}
        rd['input_dir']             = self.rd.get('data_dir_0')
        rd['output_dir']            = self.rd.get('data_dir')
        rd['geospace_env']          = geo_env
        rd['sTime']                 = self.rd.get('sTime')
        rd['eTime']                 = self.rd.get('eTime')
        rd['symh_min']              = self.rd.get('symh_min')
        rd['symh_max']              = self.rd.get('symh_max')
        rd['kp_min']                = self.rd.get('kp_min')
        rd['kp_max']                = self.rd.get('kp_max')
        lib.select_histograms.main(rd)

    def stats(self):
        # Calculate Statistics from Histograms #########################################
        rd = {}
        rd['src_dir']               = self.rd.get('data_dir')
        rd['params']                = self.rd.get('params')
        rd['xkeys']                 = self.rd.get('xkeys')
        rd['stats']                 = ['sum','mean','median','std']
        lib.statistics_histograms.main(rd)

    def baseline(self):
        # Baseline daily observations against statistics ###############################
        rd = {}
        rd['src_dir']               = self.rd.get('data_dir')
        rd['xkeys']                 = self.rd.get('xkeys')
        rd['stats']                 = ['z_score']
        lib.baseline_histograms.main(rd)

    def visualize(self):
        # Visualization ################################################################
        ### Visualize Observations
        rd = {}
        rd['srcs']                  = os.path.join(self.rd.get('data_dir'),'*.data.nc.bz2')
        rd['baseout_dir']           = self.rd.get('plot_dir')
        rd['sTime']                 = self.rd.get('sTime')
        rd['eTime']                 = self.rd.get('eTime')
        rd['geospace_env']          = self.rd.get('geo_env')
        rd['plot_kpsymh']           = self.rd.get('plot_kpsymh',True)
        rd['plot_goes']             = self.rd.get('plot_goes',True)
        rd['plot_sza']              = self.rd.get('plot_sza',True)
        rd['band_keys']             = self.rd.get('band_keys')
        rd['plot_region']           = self.rd.get('filter_region')
        rd['time_format']           = self.rd.get('time_format')
        rd['axvlines']              = self.rd.get('axvlines')
        rd['axvlines_kw']           = self.rd.get('axvlines_kw')
        rd['axvspans']              = self.rd.get('axvspans')
        lib.visualize_histograms.main(rd)
#        lib.visualize_histograms.plot_dailies(rd)

        ### Visualize Baselines
        rd['srcs']                  = os.path.join(self.rd.get('data_dir'),'*.baseline_compare.nc.bz2')
        rd['robust_dict']           = {1:False}
        lib.visualize_histograms.main(rd)
#        lib.visualize_histograms.plot_dailies(rd)

#        ### Visualize Statistics
#        rd = {}
#        rd['srcs']                  = os.path.join(self.rd.get('data_dir'),'stats.nc.bz2')
#        rd['baseout_dir']           = self.rd.get('plot_dir')
#        lib.visualize_histograms_simple.main(rd)

    def copy_stats(self,other):
        ### Copy files to the Data Directory and write a short report
        fnames  = []
        fnames.append('filetable.png')
        fnames.append('filetable.csv')
        fnames.append('stats.nc.bz2')

        results = []
        for fname in fnames:
            src = os.path.join(other.rd.get('data_dir'),fname)
            dst = os.path.join(self.rd.get('data_dir'),fname)
            shutil.copy(src,dst)

            results.append('{!s} --> {!s}\n'.format(src,dst))
        with open(os.path.join(self.rd.get('data_dir'),'stats_copy.txt'),'w') as fl:
            for line in results:
                fl.write(line)


        ### Also copy files to the Plot Directory for convenience.
        fnames  = []
        fnames.append('filetable.png')
        fnames.append('filetable.csv')

        results = []
        for fname in fnames:
            src = os.path.join(other.rd.get('data_dir'),fname)
            dst = os.path.join(self.rd.get('plot_dir'),fname)
            shutil.copy(src,dst)
            results.append('{!s} --> {!s}\n'.format(src,dst))
        with open(os.path.join(self.rd.get('plot_dir'),'stats_copy.txt'),'w') as fl:
            for line in results:
                fl.write(line)

if __name__ == '__main__':

    run_name    = 'quiet_baseline_global'
    base_dir    = 'data/histograms'
    data_dir_0  = 'data/histograms/0-10000km_dx30min_dy500km'

    rd  = {}
    rd['base_dir']      = base_dir
    rd['data_dir']      = os.path.join(base_dir,run_name)
    rd['plot_dir']      = os.path.join('output/galleries/histograms',run_name)
    rd['sTime']         = datetime.datetime(2016,1,1)
    rd['eTime']         = datetime.datetime(2018,1,1)
    rd['rgc_lim']       = (0,10000)
    rd['filter_region'] = None

    rd['reprocess']     = False

    rd['symh_min']      = -25
    rd['symh_max']      =  25
    rd['kp_min']        = None
    rd['kp_max']        =   3

    rd['plot_goes']     = False
    rd['plot_sza']      = False

    baseline = HamDataSet(rd)
    baseline.create()
#    baseline.select()
#    baseline.stats()
#    baseline.visualize()

    run_name    = 'geomag_storm_global'
    rd  = {}
    rd['data_dir_0']    = baseline.rd.get('data_dir_0')
    rd['data_dir']      = os.path.join(base_dir,run_name)
    rd['plot_dir']      = os.path.join('output/galleries/histograms',run_name)
    rd['sTime']         = datetime.datetime(2017,9,7)
    rd['eTime']         = datetime.datetime(2017,9,14)
    rd['plot_sza']      = False
    rd['time_format']   = {'format':'%d %b','rotation':0,'ha':'center','label':'Date [UT]'}

    axv         = []
    axvspans    = []
    axv.append(datetime.datetime(2017,9,7,21))
#    axv.append(datetime.datetime(2017,9,9))
    axv.append(datetime.datetime(2017,9,9,14))
    axvspans.append((axv[0],axv[1]))

#    axv.append(datetime.datetime(2017,9,12,18))
#    axv.append(datetime.datetime(2017,9,13,3))
#    axvspans.append((axv[2],axv[3]))

    rd['axvlines']      = axv
#    rd['axvlines_kw']   = {'label_time':False}
    rd['axvspans']      = axvspans

    lib.gl.prep_output({0:rd['data_dir']},clear=True)
    lib.gl.prep_output({0:rd['plot_dir']},clear=True)

    event = HamDataSet(rd)
    event.select()
    event.copy_stats(baseline)
    event.baseline()
    event.visualize()

    import ipdb; ipdb.set_trace()
