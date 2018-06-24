import datetime
from datetime import timedelta, date

def daterange(sTime,eTime):
    """ Get every date in the range start_date -> end_date (inc) """
    start_date  = datetime.datetime(sTime.year,sTime.month,sTime.day) 
    end_date    = datetime.datetime(eTime.year,eTime.month,eTime.day) 
    for n in range(int ((end_date - start_date).days)+1):
        yield start_date + timedelta(n)

def dt64_to_ut_hours(dt64):
    """ Get UT hours in a decimal from datetime64 """
    sec = dt64.astype("M8[s]")
    return sec.astype(int) % (24 * 3600) / 3600

def ut_hours_to_slt(ut_hours, long):
    """ Get SLT at a longitude from UT hours """
    return (ut_hrs + long/15.) % 24
