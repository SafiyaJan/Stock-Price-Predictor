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

if __name__ == "__main__":

	# 1. Load your saved model
	model = keras.models.load_model("models/20868193_RNN_model")


	# 2. Load your testing data and sort data by the date
	df_test = pd.read_csv('data/test_data_RNN.csv', sep=',')
	df_test['Date'] = pd.to_datetime(df_test.Date)
	df_test = df_test.sort_values(by='Date')
	date = df_test['Date']
	df_test = df_test.drop('Date',1).to_numpy()

	# split testing data into features and labels
	test_data = df_test[:,0:-1]
	test_label = df_test[:,[-1]]

	# scale data and labels values to be between 0 and 1
	test_data_scalar = MinMaxScaler()
	test_label_scalar = MinMaxScaler()

	test_data = test_data_scalar.fit_transform(test_data)
	test_label = test_label_scalar.fit_transform(test_label)


	# 3. Run prediction on the test data and output required plot and loss

	# reshape test data to feed into model
	test_data = test_data.reshape(test_data.shape[0],1,test_data.shape[1])

	# run predictions and compute MSE between Predictions and Target Values on 
	# normalized data 
	prediction = model.predict(test_data)
	testScore = mean_squared_error(test_label,prediction)

	print ("Loss (MSE) on Normalized Test Set - ",testScore)

	# compute MSE between Predictions and Target Values on denormalized data 
	testScore = mean_squared_error(test_label_scalar.inverse_transform(test_label),test_label_scalar.inverse_transform(prediction))
	
	print ("Loss (MSE) on Denormalized Test Set - ",testScore)

	# plot the actual and predicted stock prices
	plt.plot(date,test_label_scalar.inverse_transform(prediction))
	plt.plot(date,test_label_scalar.inverse_transform(test_label))
	plt.legend(['Predicted Price', 'Actual Price'])
	plt.ylabel('Stock Price ($)')
	plt.xlabel('Year')
	plt.title('Predicted v/s Actual Stock Price')
	plt.grid()
	plt.show()


