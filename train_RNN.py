# import required packages
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import keras
import pickle

# YOUR IMPLEMENTATION
# Thoroughly comment your code to make it easy to follow

def create_dataset():
	
	# retrieve data from csv file as a Dataframe and remove unwanted columns
	data_frame = pd.read_csv("data/q2_dataset.csv")
	# data_frame = data_frame.drop('Date',1)
	data_frame = data_frame.drop(' Close/Last',1)
	dates = data_frame["Date"]
	data_frame = data_frame.drop('Date',1)

	# conver data frame into numpy
	data_frame = data_frame.to_numpy()

	# create array to store features and target values of each sample/data point
	features = []
	labels = []

	# start from the earliest date
	i = len(data_frame)-1

	while (i >= 3):

		# array to gather the features from past 3 days 
		sample_features = []

		# print (i,i-1,i-2,i-3)

		# gather the features from the past 3 days 
		sample_features.append(data_frame[i].tolist())
		sample_features.append(data_frame[i-1].tolist())
		sample_features.append(data_frame[i-2].tolist())

		# store the 4th day opening price as target
		labels.append((data_frame[i-3][1]))

		# concatenate all features together to create a row of features for one sample
		sample_features = sample_features[0] + sample_features[1] + sample_features[2]

		# add features for current sample to list of all the sample
		features.append(sample_features)

		i = i - 1

	# convert to numpy arrays
	features = np.asarray(features)
	labels = np.asarray(labels).reshape(len(labels),1)

	print (len(features))
	print (len(labels))

	# append labels as an extra column
	features = np.append(features,labels,axis=1)


	# create dataframe from features arrray
	df = pd.DataFrame(data = features, columns=['Day1Volume','Day1Open',
		'Day1High','Day1Low','Day2Volume','Day2Open',
		'Day2High','Day2Low','Day3Volume','Day3Open',
		'Day3High','Day3Low', 'Target'])

	df.insert(0,"Date",dates, False)

	# split data frame into train and test set
	train, test = train_test_split(df, test_size=0.3, shuffle=True)

	# store train and test set
	train.to_csv('data/train_data_RNN.csv', index = False)
	test.to_csv('data/test_data_RNN.csv', index = False)


# scale data to have values between 0 and 1
def preprocess_data(data):

	scaler = MinMaxScaler()
	scaler.fit(data)
	# print (scaler.data_max_)
	return scaler, scaler.transform(data)


if __name__ == "__main__": 

	# create_dataset()

	# 1. load your training data

	# load data and preprocess data to have values between 0 and 1
	df_train = pd.read_csv('data/train_data_RNN.csv', sep=',').to_numpy()[:,1:]
	df_test = pd.read_csv('data/test_data_RNN.csv', sep=',').to_numpy()[:,1:]
	# print (df_train[:,1:-1].shape)

	train_scalar,preprocess_train = preprocess_data(df_train)
	test_scalar,preprocess_test = preprocess_data(df_test)
	
	train_data = preprocess_train[:,0:-1]
	train_label = preprocess_train[:,[-1]]

	test_data = preprocess_test[:,0:-1]
	test_label = preprocess_test[:,[-1]]

	train_data = train_data.reshape(train_data.shape[0],1,train_data.shape[1])
	test_data = test_data.reshape(test_data.shape[0],1,test_data.shape[1])

	# 2. Train your network
	# 		Make sure to print your training loss within training to show progress
	# 		Make sure you print the final training loss

	model = Sequential()
	model.add(LSTM(2, input_shape = (train_data.shape[1],train_data.shape[2])))
	model.add(Dense(1))
	model.compile(loss='mae',optimizer='adam')

	history = model.fit(train_data,train_label,epochs=20, verbose=1, validation_data = (test_data,test_label))

	# 3. Save your model

	model.save("models/S3JAN_model")

	plt.plot(np.asarray(history.history['loss']))
	plt.plot(np.asarray(history.history['val_loss']))
	plt.legend(['Training', 'Testing'])
	plt.xlabel('Number of Epochs')
	plt.ylabel('Loss')
	plt.title('Loss for training and testing')
	plt.grid()
	plt.show()

