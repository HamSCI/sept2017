#!/usr/bin/env python3
import os
import glob
import datetime
from collections import OrderedDict

import numpy as np
import pandas as pd

import mysql.connector

from timeutils import daterange

def load_data(db, start, finish):
    """ Loads in data from the db as dated dataframes from start to finish (exc) """

    #MariaDB [spots1a]> describe radio_spots;
    #+---------------+---------------------+------+-----+---------+-------+
    #| Field         | Type                | Null | Key | Default | Extra |
    #+---------------+---------------------+------+-----+---------+-------+
    #| rpt_key       | varchar(50)         | NO   | PRI | NULL    |       |
    #| tx            | int(10) unsigned    | NO   | MUL | NULL    |       |
    #| rx            | int(10) unsigned    | NO   |     | NULL    |       |
    #| rpt_mode      | tinyint(3) unsigned | NO   |     | NULL    |       |
    #| snr           | smallint(6)         | YES  |     | NULL    |       |
    #| freq          | float               | NO   |     | NULL    |       |
    #| occurred      | datetime            | NO   | MUL | NULL    |       |
    #| source        | tinyint(3) unsigned | NO   | MUL | NULL    |       |
    #| tx_grid       | tinytext            | NO   |     | NULL    |       |
    #| rx_grid       | tinytext            | NO   |     | NULL    |       |
    #| band          | tinyint(3) unsigned | NO   | MUL | NULL    |       |
    #| tx_lat        | float               | YES  |     | NULL    |       |
    #| tx_long       | float               | YES  |     | NULL    |       |
    #| rx_lat        | float               | YES  |     | NULL    |       |
    #| rx_long       | float               | YES  |     | NULL    |       |
    #| dist_Km       | int(11)             | YES  |     | NULL    |       |
    #| tx_mode       | tinyint(4)          | YES  |     | NULL    |       |
    #| tx_loc_source | char(1)             | YES  |     | NULL    |       |
    #| rx_loc_source | char(1)             | YES  |     | NULL    |       |
    #+---------------+---------------------+------+-----+---------+-------+

    for dt in daterange(start, finish):
        yield dt, get_df_between_times(db, dt, dt + timedelta(days=1))

def get_df_between_times(db, start, finish):
    """ Grab all radio spots between start and finish as a dataframe """
    qry = "SELECT * FROM radio_spots WHERE occurred BETWEEN %(dstart)s AND %(dfinish)s"
    df = pd.read_sql(qry, db, params = {"dstart": start, "dfinish": finish})
    return df

if __name__ == '__main__':

    in_file     = 'data/callIds/all_rgc_summary_callIds.csv.bz2'
    df          = pd.read_csv(in_file,compression='bz2')

    user        = 'hamsci'
    password    = 'hamsci'
    host        = 'localhost'
    database    = 'spots1a'
    db          = mysql.connector.connect(user=user,password=password,host=host,database=database,buffered=True)

    df['call']  = ''
    for rinx,row in df.iterrows():
        cid = row['call_ids']
        qry = 'SELECT callsign FROM radio_operators WHERE operator_id={!s};'.format(cid)

        crsr    = db.cursor()
        crsr.execute(qry)
        results = crsr.fetchall()
        crsr.close()
        
        call    = results[0][0]
        df.loc[rinx,'call'] = call
        print(rinx,call)
        if len(results) > 1:
            import ipdb; ipdb.set_trace()

    out_file     = 'data/callIds/callsigns.csv.bz2'
    df.to_csv(out_file,index=False,compression='bz2')



