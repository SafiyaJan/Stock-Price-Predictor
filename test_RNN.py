# import required packages
import pandas as pd
import numpy as np
import keras
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import math

from sklearn.metrics import mean_squared_error

# YOUR IMPLEMENTATION
# Thoroughly comment your code to make it easy to follow

# scale data to have values between 0 and 1
def preprocess_data(data):

	scaler = MinMaxScaler()
	scaler.fit(data)
	# print (scaler.data_max_)
	return scaler, scaler.transform(data)

if __name__ == "__main__":

	# 1. Load your saved model
	model = keras.models.load_model("models/S3JAN_model")

	# 2. Load your testing data
	df_test = pd.read_csv('data/test_data_RNN.csv', sep=',')

	df_test['Date'] = pd.to_datetime(df_test.Date)

	df_test = df_test.sort_values(by='Date')

	date = df_test['Date']

	df_test = df_test.drop('Date',1).to_numpy()


	test_data = df_test[:,0:-1]
	test_label = df_test[:,[-1]]

	test_data_scalar,test_data = preprocess_data(test_data)
	test_label_scalar,test_label = preprocess_data(test_label)

	# print (test_data)

	test_data = test_data.reshape(test_data.shape[0],1,test_data.shape[1])

	# 3. Run prediction on the test data and output required plot and loss

	prediction = model.predict(test_data)

	testScore = math.sqrt(mean_squared_error(test_label_scalar.inverse_transform(test_label), test_label_scalar.inverse_transform(prediction)))
	
	# print ("RMSE - ",testScore)

	plt.plot(date,test_label_scalar.inverse_transform(prediction))
	plt.plot(date,test_label_scalar.inverse_transform(test_label))
	plt.legend(['Predicted', 'Actual Values'])
	plt.ylabel('Stock Price')
	plt.show()


