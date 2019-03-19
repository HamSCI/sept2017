import datetime
import schedule
import time
import histograms

dt = datetime.date.today()

def execute_daily():
	histograms.histogram_workflow(dt.year, dt.month, dt.day)


schedule.every().day.do(execute_daily)

while True:
	schedule.run_pending()
	time.sleep(1)
