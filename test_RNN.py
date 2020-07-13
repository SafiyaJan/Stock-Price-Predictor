# import required packages
import pandas as pd

# YOUR IMPLEMENTATION
# Thoroughly comment your code to make it easy to follow

def create_dataset():
	data_frame = pd.read_csv("data/q2_dataset.csv").to_numpy()

	# print (data_frame[0:10],"\n")

	features = []
	labels = []

	i = 0
	while (i < len(data_frame)-4):
		features.append(data_frame[i])
		features.append(data_frame[i+1])
		features.append(data_frame[i+2])
		labels.append((data_frame[i+3]))
		i+=4

	print (len(features))
	
	# labels = data_frame[3::4]

	print (len(labels))



	


if __name__ == "__main__":

	create_dataset()

	# 1. Load your saved model

	# 2. Load your testing data

	# 3. Run prediction on the test data and output required plot and loss


