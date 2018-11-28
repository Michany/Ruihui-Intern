import datetime
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from data_reader import get_muti_close_day, get_index_day
TODAY = datetime.date.today().strftime('%Y-%m-%d')

#dataset_train = pd.read_csv('NSE-TATAGLOBAL.csv')
#training_set = dataset_train.iloc[:, 1:2].values
hs300 = get_index_day('000300.SH', '2007-02-01', TODAY, '1D')
training_set, testing_set = train_test_split(hs300, test_size=0.1, shuffle=False)

# 处理数据
training_set = training_set.sclose.values
training_set = training_set.reshape(-1, 1) 
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)

#%% 数据集构造
X_train = []
y_train = []
step = 60
for i in range(step, len(training_set)):
   X_train.append(training_set_scaled[i-step:i, 0])
   y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

#%% LSTM模型
regressor = Sequential()

regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))

regressor.add(Dense(units = 1))

regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

#%% 训练
regressor.fit(X_train, y_train, epochs = 100, batch_size = 32)

#%% 测试
real_stock_price = testing_set.sclose.values.reshape(-1,1)
dataset_total = hs300.sclose
inputs = dataset_total[len(dataset_total) - len(testing_set) - 60:].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)
X_test = []
for i in range(step, len(testing_set)+60):
   X_test.append(inputs[i-60:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

#%% 画图
plt.figure(figsize=(12,8))
plt.plot(real_stock_price, color = 'black', label = 'Stock Price')
plt.plot(predicted_stock_price, color = 'green', label = 'Predicted Stock Price')
plt.title('Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()
