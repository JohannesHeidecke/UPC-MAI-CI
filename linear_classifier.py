from matplotlib.pyplot import plot as plt
from matplotlib.pyplot import show, draw
import numpy as np
import os
import sys
from scipy import ndimage
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, classification_report

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
                image_data = (image_data - pixel_depth)/pixel_depth
                
                if image_data.shape != (image_size, image_size):
                    raise Exception('Unexpected Image size: %s'%str(image_data.shape))
                
                dataset[image_index,:,:] = image_data
                labels[image_index] = label_index
                image_index += 1
            
            except IOError as e:
                print('Could not read image: ', image_file," : ", e, '- Skipping..')
            
        label_index += 1
    num_images = image_index
    
    dataset = dataset[0:num_images,:,:]
    labels = labels[0:num_images]
    
    if num_images < min_images:
        raise Exception('Fewer images than expected: %d <= %d'%(num_images, min_images))
    
    print ('Dataset tensor: ', dataset.shape)
    print ('Mean: ', np.mean(dataset))
    print ('Standard Deviation: ', np.std(dataset))
    print ('Labels:', labels.shape)
    
    return dataset, labels


def randomize(dataset, labels):
    np.random.seed(42) # cos that's the answer to everything
    permutation = np.random.permutation(labels.shape[0])
    shuffled_dataset = dataset[permutation,:,:]
    shuffled_labels = labels[permutation]
    return shuffled_dataset, shuffled_labels


def results(model, valid_dataset, valid_labels, train_dataset, train_labels):
    labels = ['6', '8'] #WARNING! change this manually if you test other than 6s and 8s
    n_predict = 1000    #change this manually as well.
    X_val = valid_dataset[:n_predict].reshape(-1, valid_dataset.shape[1]*train_dataset.shape[2])
    y_val = valid_labels[:n_predict]
    print("prediction results for: ",X_val.shape, y_val.shape)
    y_pred = model.predict(X_val)
    
    print ("Score: ", classification_report(y_pred, y_val, target_names=labels))    
    plt.pcolor(confusion_matrix(y_pred, y_val), cmap="Reds")
    draw()
    
    n_vis = 10
    n_cols = 5
    n_rows = n_vis/ n_cols
    idx = np.random.randint(valid_dataset.shape[0], size=n_vis)
    X_vis = valid_dataset[idx].reshape(-1, valid_dataset.shape[1]*valid_dataset.shape[2])
    y_vis = valid_labels[idx]
    y_pred = model.predict(X_vis)
    
    fig, ax = plt.subplots(n_rows, n_cols, sharex=True, sharey=True, figsize=(n_rows, n_cols))
    fig.set_size_inches(10*n_rows, 5*n_cols)
    for i, axi in enumerate(ax.flatten()):
        axi.pcolor(X_vis[i].reshape(valid_dataset.shape[1], valid_dataset.shape[2]), cmap="Blues")
        axi.set_title("True: %s, Predicted: %s" % (labels[y_vis[i]], labels[y_pred[i]]))
    draw()
    show()
    return

train_folders, test_folders = get_folder_names()

#warning: max_images should be > sum(no.of all images in all folders)
train_dataset, train_labels = load(train_folders, 100, 5500) 
test_dataset, test_labels = load(test_folders, 100, 2500)

#randomize the dataset and labels
train_dataset, train_labels = randomize(train_dataset, train_labels)
test_dataset, test_labels = randomize(test_dataset, test_labels)


train_size = 2000
valid_size = 300

valid_dataset = train_dataset[:valid_size,:,:]
valid_labels = train_labels[:valid_size]
train_dataset = train_dataset[valid_size:valid_size+train_size,:,:]
train_labels = train_labels[valid_size:valid_size+train_size]
print ('Training', train_dataset.shape, train_labels.shape)
print ('Validation', valid_dataset.shape, valid_labels.shape)

#print(train_labels[11])
#plt.imshow(train_dataset[np.random.randint(train_dataset.shape[0])])
#plt.imshow(train_dataset[11])
#plt.show()

#convert 3D array into 2D array (vector of vectors or tensors)
n_train = -1
X_train = train_dataset[:n_train].reshape(-1, train_dataset.shape[1]*train_dataset.shape[2])
y_train = train_labels[:n_train]

print(X_train.shape) #(data_size, 784)

model = LogisticRegression(multi_class="multinomial", solver="lbfgs")
print(model.fit(X_train, y_train))

results(model, valid_dataset, valid_labels, train_dataset, train_labels)
