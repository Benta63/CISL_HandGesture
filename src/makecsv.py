#Given different directories, writes the name of each image file along with
#their label into a text file delimited by ',s'


import os
import sys
import csv

import pandas as pd
if __name__ == '__main__':
	root_dir="C:\\Users\\Noah\\Documents\\CISL\\PicturesEdited"

	train_dir="C:\\Users\\Noah\\Documents\\CISL\\Training"
	test_dir="C:\\Users\\Noah\\Documents\\CISL\\Testing"
	noise_dir="C:\\Users\\Noah\\Documents\\CISL\\Noise_Results"

	all_file_names = [name for root, dirs, files in os.walk(root_dir) for name in files]
	#The file naming convention is NUM=#
	labels = [int(file[4]) for file in all_file_names]


	with open('all_pics.txt', mode='w') as csv_file:

		csv_writer = csv.writer(csv_file, delimiter=',')
		for i in range(0, len(all_file_names)):
			csv_writer.writerow([all_file_names[i], labels[i]])

	train = pd.read_csv('tr.txt', sep=",", header=None)

	all_file_names = [name for root, dirs, files in os.walk(train_dir) for name in files]
	#The file naming convention is NUM=#
	labels = [int(file[4]) for file in all_file_names]


	with open('training.txt', mode='w') as csv_file:

		csv_writer = csv.writer(csv_file, delimiter=',')
		for i in range(0, len(all_file_names)):
			csv_writer.writerow([all_file_names[i], labels[i]])


	all_file_names = [name for root, dirs, files in os.walk(test_dir) for name in files]
	#The file naming convention is NUM=#
	labels = [int(file[4]) for file in all_file_names]


	with open('testing.txt', mode='w') as csv_file:

		csv_writer = csv.writer(csv_file, delimiter=',')
		for i in range(0, len(all_file_names)):
			csv_writer.writerow([all_file_names[i], labels[i]])

	all_file_names = [name for root, dirs, files in os.walk(noise_dir) for name in files]
	#The file naming convention is NUM=#
	labels = [int(file[4]) for file in all_file_names]


	with open('validation.txt', mode='w') as csv_file:

		csv_writer = csv.writer(csv_file, delimiter=',')
		for i in range(0, len(all_file_names)):
			csv_writer.writerow([all_file_names[i], labels[i]])


	