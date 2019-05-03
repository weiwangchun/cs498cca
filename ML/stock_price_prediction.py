
"""

Test Market Making Strategy on German Bourse


"""

import pandas as pd
import numpy as np
from datetime import date, timedelta
import requests
import csv


import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt




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



# Get "processed" features and build training set
min_lookback_length = 10 
test_length = 2

training_dates = unique_date[min_lookback_length: (len(unique_date) - test_length) ]
test_dates = unique_date[ (len(unique_date) - test_length) : len(unique_date)  ]


# get percentiles
def get_percentiles(tmp_data, ISIN, lookback = 10):
    # yesterday's value
    yesterday_value = tmp_data[-1:][ISIN][0]
    # total sum value over lookback period
    total_value = tmp_data[-lookback:][ISIN].sum()
    # relative to historical values 
    hist_rank = tmp_data[ISIN].rank()
    hist_pct = hist_rank[hist_rank.index.max()] / hist_rank.shape[0]
    # relative to peers values
    peer_rank = tmp_data[-1:].transpose().rank()
    peer_pct = peer_rank.loc[ISIN][0] / peer_rank.shape[0]

    return yesterday_value, total_value, hist_pct, peer_pct

# Train set
# placeholder for X and y variables 
X = np.zeros((len(training_dates) * len(unique_ISINs)  , 4 * 5)) 
Y = np.zeros((len(training_dates) * len(unique_ISINs)  , 1))

row_counter = 0
for date in training_dates:
    for ISIN in unique_ISINs:

        """
        Return Features 
        1) Yesterday's returns
        2) Overall returns in lookback length
        3) Yesterday's return as a percentile compared to the stock's historical return performance
        4) Yesterday's return as a percentile compared to peer return performance
        """
        tmp = ret_matrix[ ret_matrix.index.isin(unique_date[unique_date < date])]
        X[row_counter][0:4] = get_percentiles(tmp, ISIN)

        # Volume Features
        tmp = volume_matrix[ volume_matrix.index.isin(unique_date[unique_date < date])]
        X[row_counter][4:8] = get_percentiles(tmp, ISIN)
        
        # Number of Trades Features
        tmp = trades_matrix[ trades_matrix.index.isin(unique_date[unique_date < date])]
        X[row_counter][8:12] = get_percentiles(tmp, ISIN)
        
        # Return Max Features  (proxy for vol)
        tmp = retmax_matrix[ retmax_matrix.index.isin(unique_date[unique_date < date])]
        X[row_counter][12:16] = get_percentiles(tmp, ISIN)

        # Order Imbalance Features
        tmp = orderimbalance_matrix[ orderimbalance_matrix.index.isin(unique_date[unique_date < date])]
        X[row_counter][16:20] = get_percentiles(tmp, ISIN)

        # actual returns today
        ret = ret_matrix[ret_matrix.index.isin(unique_date[unique_date == date])][ISIN][0]
        if ret > 0.01:
            Y[row_counter] = 1  # positive
        elif ret < -0.01:
            Y[row_counter] = 2  # negative
        else:
            Y[row_counter] = 0  # neutral

        print(ISIN +" "+ str(date)+ " training features populated")
        row_counter += 1


# Test set   
# placeholder for X and y variables 
X_test = np.zeros((len(test_dates) * len(unique_ISINs)  , 4 * 5)) 
Y_test = np.zeros((len(test_dates) * len(unique_ISINs)  , 1))

row_counter = 0
for date in test_dates:
    for ISIN in unique_ISINs:

        """
        Return Features 
        1) Yesterday's returns
        2) Overall returns in lookback length
        3) Yesterday's return as a percentile compared to the stock's historical return performance
        4) Yesterday's return as a percentile compared to peer return performance
        """
        tmp = ret_matrix[ ret_matrix.index.isin(unique_date[unique_date < date])]
        X_test[row_counter][0:4] = get_percentiles(tmp, ISIN)

        # Volume Features
        tmp = volume_matrix[ volume_matrix.index.isin(unique_date[unique_date < date])]
        X_test[row_counter][4:8] = get_percentiles(tmp, ISIN)
        
        # Number of Trades Features
        tmp = trades_matrix[ trades_matrix.index.isin(unique_date[unique_date < date])]
        X_test[row_counter][8:12] = get_percentiles(tmp, ISIN)
        
        # Return Max Features  (proxy for vol)
        tmp = retmax_matrix[ retmax_matrix.index.isin(unique_date[unique_date < date])]
        X_test[row_counter][12:16] = get_percentiles(tmp, ISIN)

        # Order Imbalance Features
        tmp = orderimbalance_matrix[ orderimbalance_matrix.index.isin(unique_date[unique_date < date])]
        X_test[row_counter][16:20] = get_percentiles(tmp, ISIN)

        # actual returns today
        ret = ret_matrix[ret_matrix.index.isin(unique_date[unique_date == date])][ISIN][0]
        if ret > 0.01:
            Y_test[row_counter] = 1  # positive
        elif ret < -0.01:
            Y_test[row_counter] = 2  # negative
        else:
            Y_test[row_counter] = 0  # neutral

        print(ISIN +" "+ str(date)+ " test features populated")
        row_counter += 1



# Put data into dataloder
train_data = []
for i in range(len(X)):
    train_data.append([X[i], Y[i]])

test_data = []
for i in range(len(X_test)):
    test_data.append([X_test[i], Y_test[i]])




# params
params = {'batch_size': 8, 'shuffle': True}
train_loader = DataLoader(train_data, **params)
test_loader = DataLoader(test_data, **params)


def predicted_y_fn(outputs):
    predicted_y = torch.zeros(batch_size, dtype = torch.long)
    for i in range(outputs.shape[0]):
        _, predicted_y[i] = torch.max(outputs[i], 0)
    return predicted_y



def train(model,train_loader,test_loader,loss_func,opt,num_epochs=10):
    
    counter = 0
    for epoch in range(num_epochs):

        running_loss = 0.0
        for i, data in enumerate(train_loader):
            inputs, labels = data
            opt.zero_grad()

            outputs = model(inputs.to(dtype=torch.int64))
            loss = loss_func(outputs, labels)
            loss.backward()
            opt.step()

            # log training loss and training accuracy on each batch
            predicted_y = predicted_y_fn(outputs)
            train_accuracy = sum(predicted_y == labels).item() / batch_size   
            counter += 1
            writer.add_scalar('training_loss', loss.item(), counter)
            writer.add_scalar('training_accuracy', train_accuracy, counter)

            running_loss += loss.item()
        print("Epoch {} - total training loss: {}".format(epoch, running_loss))

        # after each epoch, evaluation mode
        total_loss = 0.0
        total_correct = 0.0
        for i, data in enumerate(test_loader):
            inputs, labels = data
            outputs = model(inputs.to(dtype = torch.int64))
            loss = loss_func(outputs, labels)
            # test accuracy
            predicted_y = predicted_y_fn(outputs)
            num_correct = sum(predicted_y == labels).item() 

            total_loss += loss.item()
            total_correct += num_correct
        
        total_accuracy = total_correct / 10000
        print("Epoch {} - total validation loss: {} - total validation accuracy: {}".
              format(epoch, total_loss, total_accuracy))
        



# Train Basic Neural Network
# ------------------------------------------------------------------
class TwoLayerModel(nn.Module):
  def __init__(self):
    super(TwoLayerModel, self).__init__()
    self.net = nn.Sequential(
      nn.Linear(20, 10), 
      nn.ReLU(), 
      nn.Linear(10, 3))
    
  def forward(self, x):
    return self.net(x)


model = TwoLayerModel()
loss = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr= 0.001)


train(model, train_loader, test_loader, loss, optimizer, 15)
