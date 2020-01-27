import matplotlib.pyplot as plt
import pandas as pd

import numpy as np
import tensorflow as tf
import random as rn
import os

os.environ['PYTHONHASHSEED'] = '0'

# Setting the seed for numpy-generated random numbers
np.random.seed(37)

# Setting the seed for python random numbers
rn.seed(1254)

# Setting the graph-level random seed.
tf.set_random_seed(89)

from tensorflow.python.keras import backend as K

session_conf = tf.ConfigProto(
      intra_op_parallelism_threads=4,
      inter_op_parallelism_threads=4)

#Force Tensorflow to use a single thread
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)

K.set_session(sess)

# Rest of the code follows from here on ...


from sklearn.preprocessing import MinMaxScaler
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import LSTM
from tensorflow.python.keras.layers import Dropout



#load the data
raw_data = pd.read_csv("new_data.csv", names=["openTime", "closeTime", "baseAsset", "quoteAsset", "open", "high", "low", "close", "volume", "quoteAssetVolume",	"numberOfTrades", "takerBuyBaseAssetVolume", "takerBuyQuoteAssetVolume", "takerBuyBaseAssetChange",	"takerBuyQuoteAssetChange",	"volumeChange",	"quoteAssetVolumeChange", "openChange",	"highChange", "lowChange", "closeChange", "numberOfTradesChange"]);

# extract data we want
required_data = raw_data[["open", "high", "low", "close", "volume", "quoteAssetVolume", "numberOfTrades", "takerBuyBaseAssetVolume", "takerBuyQuoteAssetVolume"]];

#init the scaler
scaler = MinMaxScaler(feature_range=(0,1));
result_scaler = MinMaxScaler(feature_range=(0,1))

#scale the extracted data
required_data_scaled = scaler.fit_transform(required_data);
request_result_scaled = result_scaler.fit_transform(np.array(required_data["close"]).reshape(-1,1))

#transform the data
features_set = []
label_set = []
prediciton_time_set = []
for i in range(0, required_data_scaled.shape[0] - 72, 72):
    features_set.append(required_data_scaled[i:i+72])
    label_set.append(request_result_scaled[i+72+24, 0]);
    prediciton_time_set.append(request_result_scaled[i + 72, 0]);
x, y, z = np.array(features_set), np.array(label_set), np.array(prediciton_time_set)

#split the data into training and test
train_quantity = 48

test_x = np.split(x, [train_quantity])[0]
test_y = np.split(y, [train_quantity])[0]
test_z = np.split(z, [train_quantity])[0]

train_x = np.split(x, [train_quantity])[1]
train_y = np.split(y, [train_quantity])[1]
train_z = np.split(z, [train_quantity])[1]


#create the model
model = Sequential()

#add the layers
model.add(LSTM(units=50, return_sequences=True, input_shape=(train_x.shape[1], train_x.shape[2])))
model.add(Dropout(0.2))

model.add(LSTM(units=100))
model.add(Dropout(0.2))

model.add(Dense(units = 1))

#compile and fit
model.compile(optimizer = 'adam', loss = 'mae')

history = model.fit(train_x, train_y, epochs = 5, batch_size = 10, validation_data=(test_x, test_y), verbose=2, shuffle=False)

#plot the training results
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show()

#make a prediction
predictions = model.predict(test_x)
normalizsed_predictions = result_scaler.inverse_transform(predictions)
normalized_test_values = result_scaler.inverse_transform(test_y.reshape(-1, 1))
normalized_prediction_times = result_scaler.inverse_transform(test_z.reshape(-1, 1))

#plot the test label(the actual price) against the predicted price
plt.figure(figsize=(10,6))
plt.plot(normalized_test_values, color='blue', label='Actual Bitcoin Price')
plt.plot(normalizsed_predictions, color='red', label='Predicted Bitcoin Price')
plt.plot(normalized_prediction_times, color='green', label='Price at Prediction')
plt.title('Bitcoin Price Prediction')
plt.xlabel('Date')
plt.ylabel('Bitcoin Price')
plt.legend()
plt.show()

correct = 0
incorrect = 0
i = 0;
while i < normalized_test_values.shape[0]:
    if normalized_test_values[i][0] >= normalized_prediction_times[i][0]:
        if normalizsed_predictions[i][0] >= normalized_prediction_times[i][0]:
            correct += 1
        else:
            incorrect += 1
    else:
        if normalizsed_predictions[i][0] <= normalized_prediction_times[i][0]:
            correct += 1
        else:
            incorrect += 1
    i += 1

print("incorrect quantity:" + str(incorrect))
print("correct quantity:" + str(correct))
print("correct results:" + str(correct/normalized_test_values.shape[0] * 100));









#IGNORE THIS -- MICHAEL TESTING
#load the data
raw_data = pd.read_csv("last_three_days.csv", names=["openTime", "closeTime", "baseAsset", "quoteAsset", "open", "high", "low", "close", "volume", "quoteAssetVolume",	"numberOfTrades", "takerBuyBaseAssetVolume", "takerBuyQuoteAssetVolume"]);

# extract data we want
required_data = raw_data[["open", "high", "low", "close", "volume", "quoteAssetVolume", "numberOfTrades", "takerBuyBaseAssetVolume", "takerBuyQuoteAssetVolume"]];

#init the scaler
scaler = MinMaxScaler(feature_range=(0,1));
result_scaler = MinMaxScaler(feature_range=(0,1))

#scale the extracted data
required_data_scaled = scaler.fit_transform(required_data);
request_result_scaled = result_scaler.fit_transform(np.array(required_data["close"]).reshape(-1,1))

#transform the data
features_set = []
label_set = []
features_set.append(required_data_scaled[0:72])
x = np.array(features_set)

predictions = model.predict(x)
normalizsed_predictions = result_scaler.inverse_transform(predictions)

print(normalizsed_predictions)

