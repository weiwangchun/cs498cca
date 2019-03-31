import os
import logging
import pandas as pd
from PySparkStock.download_deutsche import download_data
from PySparkStock.PythonHelpers.s3_utility import S3Utility



s3_utility = S3Utility()
s3_utility.s3_client.create_bucket(Bucket='emr.bucket.test.capacity')
list_files = download_data('2018-07-03')

pd.set_option('display.max_columns', 500)
df = pd.read_csv(list_files[0])
df.head()

# upload to S3
if len(list_files):
    for file_i in list_files:
        df = pd.read_csv(file_i)
        if len(pd.read_csv(file_i)) == 0:
            continue
        else:
            logging.debug(df.head())
        s3_key = os.path.split(file_i)[1]
        s3_utility.put(file_i, 'emr.bucket.test.capacity', s3_key)
