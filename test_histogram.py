#!/usr/bin/python3

import matplotlib
matplotlib.use('Agg')

import numpy as np
import pandas as pd
import cartopy.crs as ccrs
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import sys
import time

import geopack
import json

CONFIG_FILE = ""
BIN_SCALE = 1.




config = get_config(CONFIG_FILE)
prmd = setup_prmd(config)

# I'm going to explain this here in the event someone else comes upon
# this file before I'm done with it.  I'm trying out a different config
# setup inspired by rx_superposed_epoch.py which is meant to make it
# easier for non-programmers to generate graphs while also keeping
# configuration data away from sourcecode for readibility purposes
#
# I will probably end up changing this around as needed, but for now
# this is how I'm doing it :)

def get_config(config_file):
    """ Quick and dirty loader function """
    with open(config_file, "r") as f:
        config = json.load(f)

def setup_prmd(config):
    """ Creates a parameter dictionary from the JSON config """
    prmd = []
    for i, j in config["histograms"]:
        if "lim" in j:
            j["lim"] = tuple(j["lim"])
        if "bin_size" in j:
            j["bin_size"] *= BIN_SCALE
            j["bins"] = get_bins(j["lim"], j["bin_size"])
        if "cmap" in j:
            j["cmap"] = matplotlib.cm.get_cmap(j["cmap"])
        prmd[i] = j
    return prmd

def get_bins(lim, bin_size):
    """ Helper function to split a limit into bins of the proper size """
    return np.arrange(lim[0], lim[1]+2*bin_size, bin_size)


if __name__ == "__main__":
    pass
    # x-axis: UTC
    xbins = get_bins((0, 24), 10./60)
    # y-axis: distance (km)
    ybins = get_bins((0, 3000), 500)
