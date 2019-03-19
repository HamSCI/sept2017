from subprocess import *
import histograms
import datetime

#month = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
#month[0] = 31
#month[1] = 28
#month[2] = 31
#month[3] = 30
#month[4] = 31
#month[5] = 30
#month[6] = 31
#month[7] = 31
#month[8] = 30
#month[9] = 31
#month[10] = 30
#month[11] = 31

#for mon in range(1,13):
#	total_days = month[mon] + 1
#	for day in range(1,total_days):
#		run(['python3 histograms.py', mon_str, day_str], shell=True)

sDate = datetime.date(2017,4,3)
eDate = datetime.date.today()

while (sDate < eDate):
	histograms.histogram_workflow(sDate.year, sDate.month, sDate.day)
	sDate = sDate + datetime.timedelta(days=1)


		

