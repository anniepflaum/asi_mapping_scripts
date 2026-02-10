# scheduler to automatically generate realtime maps ever couple minutes

import schedule
import datetime as dt
import time
import os

refresh_rate = 30.


def job():
    os.system('python map_asi_realtime_restructure.py')
    os.system('rsync -av -e "ssh -i ~/.ssh/gioptics-key" --progress ../launch_science_fast/ realtime@optics.gi.alaska.edu:/import/amisr/archive/Processed_data/GNEISS/mapped_ASI/fast')
    most_recent = sorted(os.listdir('../launch_science_fast'))[-1]
    os.system(f'scp ../launch_science_fast/{most_recent} gioptics:/var/www/html/realtime/latest/gneiss_mapped/gneiss_mapped_fast_latest.png')

job()

schedule.every(refresh_rate).seconds.do(job)

starttime = time.monotonic()
while True:
    schedule.run_pending()
    time.sleep(refresh_rate - ((time.monotonic() - starttime) % refresh_rate))
