import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from scipy import ndimage
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

image_size = 28
pixel_depth = 255.0

def get_folder_names():
    """
    Return the list of directories inside training and testing directories.
    """
    
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


def load(folders, min_images, max_images):
    
    dataset = np.ndarray(shape = (max_images, image_size, image_size), dtype=np.float32)
    
    labels = np.ndarray(shape = (max_images), dtype = np.int32)
    label_index = 0
    image_index = 0
    
    for folder in folders:
        print (folder)
        print ('Number of Images found: ',len(os.listdir(folder)))
        for image in os.listdir(folder):
            if image_index > max_images:
                raise Exception("more images than expected: %d >= %d"%(image_index, max_images))
            image_file = os.path.join(folder, image)
            try:
                image_data = ndimage.imread(image_file).astype(float)
                image_data = image_data[:,:,1]
                #image_data = (image_data - pixel_depth/2)/pixel_depth
                
                if image_data.shape != (image_size, image_size):
                    raise Exception('Unexpected Image size: %s'%str(image_data.shape))
                
                dataset[image_index,:,:] = image_data
                labels[image_index] = label_index
                image_index += 1
            
            except IOError as e:
                print('Could not read image: ', image_file," : ", e, '- Skipping..')
            
        label_index += 1
        print('-'*40)
    num_images = image_index
    
    dataset = dataset[0:num_images,:,:]
    labels = labels[0:num_images]
    
    #normalize the dataset
    dataset = np.fabs(dataset) #absolute values
    a_max = np.amax(dataset) #get max
    dataset = (dataset - np.mean(dataset))/np.std(dataset) #divide
    
    if num_images < min_images:
        raise Exception('Fewer images than expected: %d <= %d'%(num_images, min_images))
    
    print ('Dataset tensor: ', dataset.shape)
    print ('Mean: ', np.mean(dataset))
    print ('Standard Deviation: ', np.std(dataset))
    print ('Labels:', labels.shape)
    print('-'*40)
    return dataset, labels


def randomize(dataset, labels):
    np.random.seed(42) # cos that's the answer to everything
    permutation = np.random.permutation(labels.shape[0])
    shuffled_dataset = dataset[permutation,:,:]
    shuffled_labels = labels[permutation]
    return shuffled_dataset, shuffled_labels


def confusion(y_test, y_pred):
    pass
    

#==============start of program=================================

train_folders, test_folders = get_folder_names()

#warning: max_images should be > sum(no.of all images in all folders)
train_dataset, train_labels = load(train_folders, 100, 5500) 
test_dataset, test_labels = load(test_folders, 100, 2500)

#randomize the dataset and labels
train_dataset, train_labels = randomize(train_dataset, train_labels)
test_dataset, test_labels = randomize(test_dataset, test_labels)

#convert 3D array into 2D array (vector of vectors or tensors)
X_train = train_dataset.reshape(-1, train_dataset.shape[1]*train_dataset.shape[2])
y_train = train_labels

x_test = test_dataset.reshape(-1, test_dataset.shape[1]*test_dataset.shape[2])
y_test = test_labels

#print(X_train.shape) #(data_size, 784)

model = LogisticRegression(multi_class="ovr", solver="liblinear", n_jobs=-1)
model.fit(X_train, y_train)

y_pred = model.predict(x_test)

CM = confusion_matrix(y_pred, y_test)

TN = CM[0][0]
FN = CM[1][0]
TP = CM[1][1]
FP = CM[0][1]

print ("True Positive: %d True Negative %d False Postive %d False Negative %d"%(TP, TN, FP, FN))


print(model.score(x_test, y_test)*100 ,"%")
