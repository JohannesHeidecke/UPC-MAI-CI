import os
import sys
import numpy as np
from scipy import ndimage
import pickle
from sklearn import datasets, linear_model


image_size = 28
pixel_depth = 255.0

def get_folders(folder):
	folders = [x[0] for x in os.walk(folder)]
	folders.sort()
	return folders[1:]

def load_letter(folder, min_num_images):
	"""Load images from a single folder with the name same as that of the images' label
	   for example load_letter('notMNIST_small/A', 20) will load all the images in directory the /A.
	"""

	image_files = os.listdir(folder)

	dataset = np.ndarray(shape = (len(image_files), image_size, image_size), dtype=np.float64)
	#Shape of dataset here is [number_of_images_in_dir, 28, 28]
	#In this case [1873, 28, 28] for the directory notMNIST_small/A

	num_images = 0

	for image in image_files:
		#get the full path of the image
		image_file = os.path.join(folder, image)

		try:
			image_data = (ndimage.imread(image_file).astype(float) - pixel_depth/2) / pixel_depth
			
			if image_data.shape != (image_size, image_size):
				#make sure the image is 28x28 else raise exception
				raise Exception('Unexpected image shape')
			
			dataset[num_images, :,:] = image_data 
			#copy the image data into dataset
			num_images += 1
		except IOError as err:
			print("couldn't read image... skipping")
		
	#there may be some non-28x28 images so reset the length of 'datatset'
	#in this case dataset is now[1872, 28, 28] because there's only 1 corrput image in notMNIST_small/A
	dataset = dataset[0:num_images, :, :]
	
	if num_images < min_num_images:
		#make sure you have the minimum no.of proper images for the classifier to work.
		raise Exception('Many fewer images than expected: %d < %d' %(num_images, min_num_images))
			
	#print('Full dataset tensor:', dataset.shape)
	print('Mean:', np.mean(dataset))
	print('Standard deviation:', np.std(dataset))
	
	#Finally, return the entire matrix for the particular letter class.
	return dataset

def picke_it(data_folders, min_images_per_class, force=False):
	"""Store the data in images in a pickel instead of loading it in RAM in case your computer is slow"""
	dataset_names = []
	
	for folder in data_folders:
		#create a folder_name.pickle file eg: A.pickle inside the main notMNIST_small/ (or _large) directory
		set_filename = folder + '.pickle'
		dataset_names.append(set_filename)
		
		if os.path.exists(set_filename) and not force:
			print('file exists...skipping %s'% set_filename)
			#dataset_names.append(set_filename)
			pass
		else:
			#print('Pickling %s' % set_filename)
			#get the dataset for every letter from load_letter() above
			dataset = load_letter(folder, min_images_per_class)
			try:
				with open (set_filename, 'wb') as f:
					pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)
			except Exception as e:
				print("unable to save data to ", set_filename, ":", e)
		
	#return the full names of the pickle files
	return dataset_names


def make_arrays(nb_rows, img_size):
	if nb_rows:
		dataset = np.ndarray((nb_rows, img_size, img_size), dtype=np.float32)
		labels = np.ndarray(nb_rows, dtype=np.int32)
	else:
		dataset, labels = None, None
	return dataset, labels

def merge_datasets(pickle_files, train_size, valid_size=0):
	#in our example we have 10 classes A to J
	num_classes = len(pickle_files) #num_classes = 10
	valid_dataset, valid_labels = make_arrays(valid_size, image_size) #eg: data, label = ([1000, 28,28],[1000,1])
	train_dataset, train_labels = make_arrays(train_size, image_size) #eg: data,label = ([3333, 28, 28], [3333,1])
	vsize_per_class = valid_size // num_classes #vsize = 1000//10 = 100
	tsize_per_class = train_size // num_classes #tsize = 3333//10 = 333 (rounded) considering balance

	start_v, start_t = 0, 0
	end_v, end_t = vsize_per_class, tsize_per_class
	end_l = vsize_per_class+tsize_per_class
	for label, pickle_file in enumerate(pickle_files):       
		try:
			with open(pickle_file, 'rb') as f:
				letter_set = pickle.load(f)
				# let's shuffle the letters to have random validation and training set
				np.random.shuffle(letter_set)
				if valid_dataset is not None:
					valid_letter = letter_set[:vsize_per_class, :, :]
					valid_dataset[start_v:end_v, :, :] = valid_letter
					valid_labels[start_v:end_v] = label
					start_v += vsize_per_class
					end_v += vsize_per_class

				train_letter = letter_set[vsize_per_class:end_l, :, :]
				train_dataset[start_t:end_t, :, :] = train_letter
				train_labels[start_t:end_t] = label
				start_t += tsize_per_class
				end_t += tsize_per_class
		except Exception as e:
			print('Unable to process data from', pickle_file, ':', e)
			raise

	return valid_dataset, valid_labels, train_dataset, train_labels


def randomize(dataset, labels):
	permutation = np.random.permutation(labels.shape[0])
	shuffled_dataset = dataset[permutation,:,:]
	shuffled_labels = labels[permutation]
	return shuffled_dataset, shuffled_labels


def get_folder_names():
	arguments = sys.argv
	train_folders = os.path.abspath(arguments[1])
	train_folders = [os.path.abspath(x[0]) for x in os.walk(train_folders)]
	train_folders.sort()
	train_folders = train_folders[1:]

	test_folders = os.path.abspath(arguments[2])
	test_folders = [os.path.abspath(x[0]) for x in os.walk(test_folders)]
	test_folders.sort()
	test_folders = test_folders[1:]
	
	return train_folders, test_folders

#step1: get the folders for training set and testing set
train_folders, test_folders = get_folder_names()
#print(train_folders)
#print(test_folders)

#step2: read the images and store them into datasets and pickles(for later processing)
train_datasets = picke_it(train_folders, 30000) #change the values as necessary
test_datasets = picke_it(test_folders, 1000)

#step3: split dataset into training and validation as per required size.
#tain_size = int(input('Enter training size: '))
#valid_size = int(input('Enter training size: '))
#test_size = int(input('Enter training size: '))

train_size = 200000
valid_size = 10000
test_size = 10000

valid_dataset, valid_labels, train_dataset, train_labels = merge_datasets(train_datasets, train_size, valid_size) 
dummy_set, dummy_labels, test_dataset, test_labels = merge_datasets(test_datasets, test_size)


#step4: optionally, randomize the datasets
train_dataset, train_labels = randomize(train_dataset, train_labels)
test_dataset, test_labels = randomize(test_dataset, test_labels)
valid_dataset, valid_labels = randomize(valid_dataset, valid_labels)


#step5: use the sklearn.linear classifier to classify the data
#because we need to process the image as a tensor, we need to reshape the image into a vector
data = train_dataset.reshape((train_dataset.shape[0], -1))
lin_classifier = linear_model.LinearRegression()
lin_classifier.fit(data[:500], train_labels[:500])

print(lin_classifier.predict(data)[0:10])

print(train_labels[0:10]) 
