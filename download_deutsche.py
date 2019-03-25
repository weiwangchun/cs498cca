"""

Download Deutsche Bourse Dataset into csv

"""

import requests
import csv
import datetime
import os
os.chdir("C:\\Users\\uqwwei4\\Documents\\UIUC\\cs498cca\\")



start_date = '2017-08-01'
end_date = '2017-08-15'


# download minute by minute data for select date
# date format 'YYYY-MM-DD'
# -------------------------------------------
def download_data(date):
    url = 'https://s3.eu-central-1.amazonaws.com/deutsche-boerse-xetra-pds/'
    tmp_data = []
    for x in range(0, 24):
        if len(str(x)) == 1:
            filename = date + '_BINS_XETR0' + str(x) + '.csv'
        else:
            filename = date + '_BINS_XETR' + str(x) + '.csv'

        full_url = url + date + '/' + filename
        response = requests.get(full_url)
        content = response.content.decode('utf-8')
        cr = list(csv.reader(content.splitlines(), delimiter = ',')) 
        if len(cr) > 1:
            tmp_data.extend(cr)
    return tmp_data

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

    with open("sample_xetra.csv", 'w', newline = '') as f:
        w = csv.writer(f, delimiter = '\t')
        w.writerows(data)


