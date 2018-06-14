#!/usr/bin/env python3
import datetime
import numpy as np
import pandas as pd
import bz2

import matplotlib as mpl
mpl.use('Agg')
from matplotlib import pyplot as plt

pd.set_option('display.height', 1000)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

# Parameter Dict
prmd = {}

tmp = {}
tmp['title']    = "GOES 15 X-Ray (0.1-0.8 nm) Irradiance";
tmp['label']    = "W/m^2";
prmd['B_AVG']   = tmp

#float A_AVG(record);
#A_AVG:description = "XRS short wavelength channel irradiance (0.05 - 0.4 nm)";
#A_AVG:long_label = "x-ray (0.05-0.4 nm) irradiance";
#A_AVG:short_label = "xs fx";
#A_AVG:plot_label = "XS(0.05-0.4 nm)";
#A_AVG:lin_log = "log";
#A_AVG:units = "W/m^2";
#A_AVG:format = "E10.4";
#A_AVG:nominal_min = "1.0000E-09";
#A_AVG:nominal_max = "1.0000E-03";
#A_AVG:missing_value = "-99999";
#
#float B_AVG(record);
#B_AVG:description = "XRS long wavelength channel irradiance (0.1-0.8 nm)";
#B_AVG:long_label = "x-ray (0.1-0.8 nm) irradiance";
#B_AVG:short_label = "xl fx";
#B_AVG:plot_label = "XL(0.1-0.8 nm)";
#B_AVG:lin_log = "log";
#B_AVG:units = "W/m^2";
#B_AVG:format = "E10.4";
#B_AVG:nominal_min = "1.0000E-09";
#B_AVG:nominal_max = "1.0000E-03";
#B_AVG:missing_value = "-99999";
#
#// global attributes
#:conventions = "GOES Space Weather"
#:title = "GOES X-ray Sensor"
#:institution = "NOAA"
#:source = "Satellite Insitu Observations"
#:satellite_id = "GOES-15"
#:instrument = "X-ray Sensor"
#:process-type = "unknown"
#:process-level = "level 2"
#:sample-time = "1"
#:sample-unit = "minutes"
#:creation_date = "2017-10-01 10:16:19.825 UTC"
#:start_date = "2017-09-01 00:00:00.000 UTC"
#:end_date = "2017-09-30 23:59:00.000 UTC"
#:records_present = "43200"
#:originating_agency = "DOC/NOAA/NCEP/NWS/SWPC"
#:archiving_agency = "DOC/NOAA/NESDIS/NCEI"
#:NOAA_scaling_factors = "The XRS data for GOES 8-15 include NOAA scaling factors which must be removed to get observed fluxes.  To remove these factors, divide the short channel fluxes (A_FLUX) by 0.85 and divide the long channel fluxes (B_FLUX) by 0.7." 

def lines_to_df(lines,key):
    data        = False
    data_lines  = []
    for line in lines:
        if data:
            if line == '':
                break

            spl = line.split(',') 
            data_lines.append(spl)

        if key in line: data = True

    cols        = data_lines[0]
    data_lines  = data_lines[1:]

    # Create dataframe and convert date string to datetime.
    dt_key          = cols[0]
    df              = pd.DataFrame(data_lines,columns=cols)
    df['datetime']  = pd.to_datetime(df[dt_key])
    df              = df.set_index('datetime')
    del df[dt_key]

    for key in df.keys():
        df[key] = df[key].astype(np.float)

    return df

def load_goes_xrs(src = 'data/goes15_xrs/g15_xrs_1m_20170901_20170930.csv.bz2'):
    # Load data from CSV.
    with bz2.BZ2File(src) as fl:
        lines   = fl.readlines()
    lines   = [x.decode('utf-8').strip('\r\n') for x in lines]

    # Convert into dataframes.
    loc_df              = lines_to_df(lines,'satellite location:')
    loc_df['longitude'] = -1*loc_df['west_longitude']
    del loc_df['west_longitude']

    df      = lines_to_df(lines,'data:')
    df      = pd.concat([df,loc_df],axis=1)

    for key in loc_df.keys():
        df[key].fillna(method='backfill',inplace=True)

    import ipdb; ipdb.set_trace()
    return df

if __name__ == '__main__':
    df  = load_goes_xrs()
    import ipdb; ipdb.set_trace()
