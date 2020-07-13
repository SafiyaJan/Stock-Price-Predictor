# import required packages
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# YOUR IMPLEMENTATION
# Thoroughly comment your code to make it easy to follow

def create_dataset():
	
	# retrieve data from csv file as a Dataframe and remove unwanted columns
	data_frame = pd.read_csv("data/q2_dataset.csv")
	data_frame = data_frame.drop('Date',1)
	data_frame = data_frame.drop(' Close/Last',1)

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

	# append labels as an extra column
	features = (np.append(features,labels,axis=1))

	# create dataframe from features arrray
	df = pd.DataFrame(data = features, columns=['Day1Volume','Day1Open',
		'Day1High','Day1Low','Day2Volume','Day2Open',
		'Day2High','Day2Low','Day3Volume','Day3Open',
		'Day3High','Day3Low', 'Target'])


	# split data frame into train and test set
	train, test = train_test_split(df, test_size=0.3, shuffle=True)

	# store train and test set
	train.to_csv('data/train_data_RNN.csv', index = False)
	test.to_csv('data/test_data_RNN.csv', index = False)


if __name__ == "__main__": 

	create_dataset()
	# 1. load your training data

	# 2. Train your network
	# 		Make sure to print your training loss within training to show progress
	# 		Make sure you print the final training loss

	# 3. Save your model