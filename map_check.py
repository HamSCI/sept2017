#!/usr/bin/python3
import os
from collections import OrderedDict

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import cartopy.crs as ccrs

import numpy as np
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

from gen_lib import prep_output

regions = {}
tmp     = {}
tmp['lon_lim']  = (-180.,180.)
tmp['lat_lim']  = ( -90., 90.)
regions['World']    = tmp

tmp     = {}
tmp['lon_lim']  = (-130.,-60.)
tmp['lat_lim']  = (  20., 55.)
regions['US']   = tmp

tmp     = {}
tmp['lon_lim']  = ( -15., 55.)
tmp['lat_lim']  = (  30., 65.)
regions['Europe']   = tmp

tmp     = {}
tmp['lon_lim']  = ( -90.,-60.)
tmp['lat_lim']  = (  15., 30.)
regions['Carribean']    = tmp

tmp     = {}
tmp['lon_lim']  = ( -110.,-30.)
tmp['lat_lim']  = (    0., 45.)
regions['Greater Carribean']    = tmp

goess   = OrderedDict()
tmp     = {}
key     = 'GOES-EAST'
tmp['lon']      = -75
tmp['label']    = key
tmp['color']    = 'blue'
goess[key]      = tmp

tmp     = {}
key     = 'GOES-WEST'
tmp['lon']      = -135
tmp['label']    = key
tmp['color']    = 'red'
goess[key]      = tmp

def plot_map(maplim_region='World',output_dir='output',plot_goes=False):
    projection  = ccrs.PlateCarree()

    fig = plt.figure(figsize=(10,8))
    ax  = fig.add_subplot(1,1,1, projection=projection)

    ax.set_xlim(regions[maplim_region]['lon_lim'])
    ax.set_ylim(regions[maplim_region]['lat_lim'])

    ax.coastlines()
    ax.gridlines(draw_labels=True)

    if plot_goes:
        for key,item in goess.items():
            lon     = item.get('lon')
            label   = item.get('label',str(key))
            color   = item.get('color','blue')

            ax.axvline(lon,ls='--',label=label,color=color)
        ax.legend(loc='lower right')

    fname   = 'map-{!s}.png'.format(maplim_region)
    fpath   = os.path.join(output_dir,fname)
    fig.savefig(fpath,bbox_inches='tight')
    plt.close(fig)

if __name__ == '__main__':
    output_dir  = 'output/galleries/map_check'
    prep_output({0:output_dir},clear=False,php=False)


    run_dcts = []
    rd = {}
    rd['maplim_region'] = 'World'
    rd['plot_goes']     = True
    rd['output_dir']    = output_dir
    run_dcts.append(rd)

    for rd in run_dcts:
        plot_map(**rd)
