import sqlite3
import argparse
import csv

# parse file name from command line
parser = argparse.ArgumentParser()
parser.add_argument("-f", help="sqlite file")
args = parser.parse_args()


conn = sqlite3.connect(args.f)
c = conn.cursor()
c.execute('''
        select
            start,
            end,
            (cakr.end - cakr.start) as duration ,
            si.value as shortName  from CUPTI_ACTIVITY_KIND_KERNEL cakr 
        inner join 
            StringIds si 
        on si.id = cakr.shortName  
        order by `start` asc;
        ''')
results = c.fetchall()

cols = [( 'start', 'end', 'duration', 'shortName' )]
results = cols + results
results_name = args.f.replace('.sqlite', '.csv')
fp = open(f'./{results_name}', 'w')
results_file = csv.writer(fp)
results_file.writerows(results)