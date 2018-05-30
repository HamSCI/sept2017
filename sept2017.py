#!/usr/bin/env python3
import os
import glob
import datetime
from collections import OrderedDict

import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
from matplotlib import pyplot as plt

import mysql.connector

import tqdm
import seqp
from seqp import calcSun


def load_data():
    sDate       = datetime.datetime(2017,9,1)
    eDate       = datetime.datetime(2017,10,1)

    user        = 'hamsci'
    password    = 'hamsci'
    host        = 'localhost'
    database    = 'spots1a'
    db          = mysql.connector.connect(user=user,password=password,host=host,database=database,buffered=True)

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

    qry     = "SELECT * FROM radio_spots WHERE occurred BETWEEN '{!s}' AND '{!s}';".format(sDate,eDate)
               
    crsr    = db.cursor()
    crsr.execute(qry)
    results = crsr.fetchall()
    crsr.close()

    data_list   = []
    for result in results:
        import ipdb; ipdb.set_trace()

if __name__ == '__main__':

    df = load_data()

