#!/usr/bin/env python3

"""
Script covering the entire histogram workflow process.
"""
import os
import shutil
import glob
import datetime
import library as lib

geo_env     = lib.GeospaceEnv()

class HamDataSet(object):
    def __init__(self,run_dct={}):
        run_dct['params']   = run_dct.get('params',['spot_density'])
        run_dct['xkeys']    = run_dct.get('xkeys', ['ut_hrs','slt_mid'])
        self.rd = run_dct

    def create(self):
        # Create histogram NetCDF Files ################################################
        rd  = {}
        rd['sDate']                 = self.rd.get('sTime')
        rd['eDate']                 = self.rd.get('eTime')
        rd['params']                = self.rd.get('params')
        rd['xkeys']                 = xkeys
        rd['rgc_lim']               = (0,10000)
        rd['filter_region']         = None
        rd['filter_region_kind']    = 'mids'
        rd['xb_size_min']           = 30.
        rd['yb_size_km']            = 500.
        rd['output_dir']            = self.rd.get('data_dir_0')
        rd['band_obj']              = lib.gl.BandData()
        lib.calculate_histograms.main(rd)

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
        rd['srcs']                  = os.path.join(self.rd.get('data_dir'),'*.data.nc')
        rd['baseout_dir']           = self.rd.get('plot_dir')
        rd['sTime']                 = self.rd.get('sTime')
        rd['eTime']                 = self.rd.get('eTime')
        rd['geospace_env']          = self.rd.get('geo_env')
        lib.visualize_histograms.main(rd)
#        lib.visualize_histograms.plot_dailies(rd)

        ### Visualize Baselines
        rd['srcs']                  = os.path.join(self.rd.get('data_dir'),'*.baseline_compare.nc')
        rd['robust_dict']           = {1:False}
        lib.visualize_histograms.main(rd)
#        lib.visualize_histograms.plot_dailies(rd)

        ### Visualize Statistics
        rd = {}
        rd['srcs']                  = os.path.join(self.rd.get('data_dir'),'stats.nc')
        rd['baseout_dir']           = self.rd.get('plot_dir')
        lib.visualize_histograms_simple.main(rd)

    def copy_stats(self,other):
        ### Copy files to the Data Directory and write a short report
        fnames  = []
        fnames.append('filetable.png')
        fnames.append('filetable.csv')
        fnames.append('stats.nc')

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
    rd  = {}
    rd['data_dir_0'] = 'data/histograms/0-10000km_dx30min_dy500km'
    rd['data_dir']   = os.path.join('data/histograms',run_name)
    rd['plot_dir']   = os.path.join('output/galleries/histograms',run_name)
    rd['sTime']      = datetime.datetime(2016,1,1)
    rd['eTime']      = datetime.datetime(2018,1,1)

    rd['symh_min']   = -25
    rd['symh_max']   =  25
    rd['kp_min']     = None
    rd['kp_max']     =   3

    baseline = HamDataSet(rd)
#    baseline.select()
#    baseline.stats()
#    baseline.visualize()

    run_name    = 'geomag_storm_global'
    rd  = {}
    rd['data_dir_0'] = 'data/histograms/0-10000km_dx30min_dy500km'
    rd['data_dir']   = os.path.join('data/histograms',run_name)
    rd['plot_dir']   = os.path.join('output/galleries/histograms',run_name)
    rd['sTime']      = datetime.datetime(2017,9,7)
    rd['eTime']      = datetime.datetime(2017,9,14)

    lib.gl.prep_output({0:rd['data_dir']},clear=True)
    lib.gl.prep_output({0:rd['plot_dir']},clear=True)

    event = HamDataSet(rd)
    event.select()
    event.copy_stats(baseline)
    event.baseline()
    event.visualize()

    import ipdb; ipdb.set_trace()
