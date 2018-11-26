from datetime import timedelta, date

def daterange(start_date, end_date):
    """ Get every date in the range start_date -> end_date (exc) """
    for n in range(int ((end_date - start_date).days)):
        yield start_date + timedelta(n)

def dt64_to_ut_hours(dt64):
    """ Get UT hours in a decimal from datetime64 """
    sec = dt64.astype("M8[s]")
    return sec.astype(int) % (24 * 3600) / 3600

def ut_hours_to_slt(ut_hours, long):
    """ Get SLT at a longitude from UT hours """
    return (ut_hrs + long/15.) % 24
