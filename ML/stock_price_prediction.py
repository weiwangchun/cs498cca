
"""

Test Market Making Strategy on German Bourse


"""

import pandas as pd
import numpy as np
from datetime import date, timedelta
import requests
import csv






# Load all the trades into one dataframe
def batch_load(start_date, end_date):
    data = []
    for i in range(delta.days + 1):
        date = start_date + timedelta(i)
        for x in range(8, 16):
            if len(str(x)) == 1:
                filename = str(date) + '_BINS_XETR0' + str(x) + '.csv'
            else:
                filename = str(date) + '_BINS_XETR' + str(x) + '.csv'

            # change this to load directly from S3 bucket
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # I'm loading it from my C drive
            try:
                tmp_data = pd.read_csv("C:\\Users\\uqwwei4\\Documents\\UIUC\\cs498cca\\Deutsche\\" + filename, sep =",")
                data.append(tmp_data)
            except:
                print("No trades on date " +  str(date) + " at " + str(x) + ":00" )

    return pd.concat(data)


# Get basic features for a select stock
def get_info(select_data):
    length = select_data.shape[0] 
    total_volume = select_data["TradedVolume"].sum()
    total_trades = select_data["NumberOfTrades"].sum()
    # hourly volume weighted average price
    vwap = np.dot(select_data["EndPrice"], select_data["TradedVolume"]) /total_volume
    # return
    total_return = select_data.iloc[length-1]["EndPrice"] / select_data.iloc[1]["StartPrice"] - 1
    max_return = abs(max(select_data["MaxPrice"]) - min(select_data["MinPrice"])) / vwap
    # order imbalance estimate
    select_data.loc[:,"Buy_Initiation"] = (select_data.loc[:,"EndPrice"] > select_data.loc[:,"StartPrice"])
    select_data.loc[:, "Buy_Initiation"] = select_data.loc[:,"Buy_Initiation"]*1
    buy_trades = np.sum(select_data["NumberOfTrades"] * select_data["Buy_Initiation"])
    sell_trades = total_trades - buy_trades
    order_imbalance = abs(buy_trades - sell_trades) / total_trades

    return total_volume, total_trades, vwap, total_return, max_return, order_imbalance




start_date = date(2018, 1, 2)
end_date = date(2018, 3, 1)  # change to this -> end_date = date(2019, 4, 2)
delta = end_date - start_date

data = batch_load(start_date, end_date)
unique_date = data['Date'].unique()
unique_ISINs = data['ISIN'].unique()


# DELETE THIS
# I'm taking the first 50 stocks for speed
unique_ISINs = unique_ISINs[0:50]
# END DELETE


# For each stock, on each day, get features
#  -------------------------------------------------------
# placeholders for return, volume, trades, max ret, vwap, order imbalance
ret_matrix = pd.DataFrame(index = unique_date, columns = unique_ISINs).fillna(0.0000) 
volume_matrix = pd.DataFrame(index = unique_date, columns = unique_ISINs).fillna(0) 
trades_matrix = pd.DataFrame(index = unique_date, columns = unique_ISINs).fillna(0) 
retmax_matrix = pd.DataFrame(index = unique_date, columns = unique_ISINs).fillna(0.0000) 
vwap_matrix = pd.DataFrame(index = unique_date, columns = unique_ISINs).fillna(0.0000) 
orderimbalance_matrix = pd.DataFrame(index = unique_date, columns = unique_ISINs).fillna(0.0000) 

for date in unique_date:
    for ISIN in unique_ISINs:
        select_data = data[(data["ISIN"] == ISIN) & (data["Date"] == str(date)) ]

        if select_data.shape[0] > 2:
            [volume, trades, vwap, ret, max_ret, orderimbalance] = get_info(select_data)

            ret_matrix[ISIN][str(date)] = ret
            volume_matrix[ISIN][str(date)] = volume
            trades_matrix[ISIN][str(date)] = trades
            retmax_matrix[ISIN][str(date)] = max_ret
            vwap_matrix[ISIN][str(date)] = vwap
            orderimbalance_matrix[ISIN][str(date)] = orderimbalance
            print(ISIN +" " + str(date) + " : Inserted")

        else:
            print(ISIN +" " + str(date) + " : No Data")









