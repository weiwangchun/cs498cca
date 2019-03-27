
"""

Download Deutsche Bourse Dataset into csv

"""

import logging
import requests
import csv
import datetime
import os
import threading

import pandas as pd

from PySparkStock.PythonHelpers import set_logging, start_multi_threading

set_logging(10)

start_date = '2017-08-01'
end_date = '2017-08-15'


# download minute by minute data for select date
# date format 'YYYY-MM-DD'
# -------------------------------------------


def _download_onehour_data(full_url, file_path):
    response = requests.get(full_url)
    content = response.content.decode('utf-8')
    
    with open(file_path, 'w') as f:
        f.write(content)


def download_data(date,
                  download_folder = '/tmp/',
                  bypass_if_exists=True,
                  s3_bucket_url = 'https://s3.eu-central-1.amazonaws.com/deutsche-boerse-xetra-pds/'):
    """
    :param date:
    :return:
    
    date = '2017-08-15'
    x = 12
    download_folder = '/tmp/'
    """
    
    list_files = []
    list_jobs = []
    for x in range(7, 16):
        if len(str(x)) == 1:
            filename = date + '_BINS_XETR0' + str(x) + '.csv'
        else:
            filename = date + '_BINS_XETR' + str(x) + '.csv'

        logging.info(filename)
        file_path = download_folder + filename
        
        if bypass_if_exists and os.path.exists(file_path):
            logging.info('Bypass as already exists')

        full_url = s3_bucket_url + date + '/' + filename

        job_i = threading.Thread(target=_download_onehour_data,
                         kwargs={'full_url':full_url,
                                 'file_path':file_path})
                
        list_jobs.append(job_i)
        list_files.append(file_path)
    
    start_multi_threading(list_jobs, max_threads=16-7)
    
    return list_files

# generate date range
# -------------------------------------------
def generate_dates(start_date, end_date):
    start = datetime.datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.datetime.strptime(end_date, "%Y-%m-%d")
    date_generated = [start + datetime.timedelta(days=x) for x in range(0, (end-start).days)]
    date_range = []
    for date in date_generated:
        date_range.extend([date.strftime("%Y-%m-%d")])
    return date_range


if __name__ == '__main__':
    dates = generate_dates(start_date, end_date)
    data = []
    for date in dates:
        tmp = download_data(date)
        data.extend(tmp)
    #
    # with open("sample_xetra.csv", 'w', newline = '') as f:
    #     w = csv.writer(f, delimiter = '\t')
    #     w.writerows(data)


