#!/usr/bin/python3
# Create plots from the csv data
# this script is very much a prototype

# fix because there's no X server
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

sources = {
    0: "dxcluster",
    1: "WSPRNet",
    2: "RBN"
}


def plot_map(ax, df, name):
    """ Make a world-map plot of the dataframe """

    tx_df = df[['tx_long', 'tx_lat']].drop_duplicates()
    print("Transmit_sites:", len(tx_df))
    print("Midpoints:", len(df))

    tx_patch = mpatches.Patch(color='red', label='Transmit Site (N=' + str(len(tx_df)) + ')')
    md_patch = mpatches.Patch(color='blue', label='Midpoint N=' + str(len(df)) + ')')

    df.plot.scatter('md_long', 'md_lat', color="blue", ax=ax, marker="o")
    tx_df.plot.scatter('tx_long', 'tx_lat', color="red", ax=ax, marker=".")

    ax.legend(handles=[tx_patch, md_patch])
    ax.set_title("Transmitter Sites (" + name + ")")


def plot_from_csv(csv_name, remove_estimates=True):
    print("Gathering data...")
    dtn = csv_name.split(".")[0].split("/")[-1]
    df = pd.read_csv(csv_name)
    print("Data gathered!")

    print("Processing...")
    start = time.time()
    # df = df.head(10000) # Take a much smaller sample set for testing
    cols = list(df) + ["md_lat", "md_long"]
    df = df.reindex(columns=cols)
    midpoints = geopack.midpoint(df["tx_lat"], df["tx_long"], df["rx_lat"], df["rx_long"])
    df['md_lat'] = midpoints[0]
    df['md_long'] = midpoints[1]

    if remove_estimates:
        # Only show confirmed values, not estimates for latlong
        df = df.loc[(df['tx_loc_source'] != 'E') & (df['rx_loc_source'] != 'E')]
    end = time.time()
    print("Processed in {} seconds!".format(str(end - start)))

    for i in range(3):
        fig = plt.figure(figsize=(25.6, 19.2))
        map_ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
        map_ax.coastlines()
        map_ax.gridlines()
        print("Generating map for " + sources[i] + "...")
        frame = df.loc[(df['source'] == i)]
        name = dtn + "-" + sources[i]
        plot_map(map_ax, frame, name)
        print("Map generated!")
        plt.savefig("maps/" + name + ".png")

    print("Done!")


if __name__ == "__main__":
    dtf = sys.argv[1]
    plot_from_csv(dtf)
