import os
import shutil
import linear_classifier
import pickle

def clearTrainFolders():
	# print('### Clearing training folders.')
	for label in LABEL_FOLDERS:
		labelPath = os.path.join(TRAIN_FOLDER, label)
		for element in os.listdir(labelPath):
			elPath = os.path.join(labelPath, element)
			if os.path.isfile(elPath):
				fName, fExtension = os.path.splitext(elPath)
				if fExtension == '.png':
					os.unlink(elPath)
	return

def fillTrainFolders(numberOfOriginal, numberOfGenerated, iteration):
	for label in LABEL_FOLDERS:
		folder = 'samples'
		folder = os.path.join(folder, (str(label) + '_' + str(numberOfOriginal) + '_' + str(iteration)))
		# copy original images:
		for imgId in range(0, numberOfOriginal):
			fileName = 'original_' + str(imgId) + '.png'
			filePath = os.path.join(folder, fileName)
			shutil.copy2(filePath, os.path.join(TRAIN_FOLDER, label))
		# copy generated images:
		for imgId in range(0, numberOfGenerated):
			fileName = 'generated_' + str(imgId) + '.png'
			filePath = os.path.join(folder, fileName)
			shutil.copy2(filePath, os.path.join(TRAIN_FOLDER, label))
	return

def saveObject(obj, filename):
    with open(filename, 'wb') as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)


# # # # # executed code starts here: # # # # #

TRAIN_FOLDER = 'training'
TEST_FOLDER = 'test'
LABEL_FOLDERS = ['6', '8']

if not os.path.exists(TRAIN_FOLDER):
	os.makedirs(TRAIN_FOLDER)
	for label in LABEL_FOLDERS:
		if not os.path.exists(os.path.join(TRAIN_FOLDER, label)):
			os.makedirs(os.path.join(TRAIN_FOLDER, label))

if not os.path.exists(TEST_FOLDER):
	os.makedirs(TEST_FOLDER)
	for label in LABEL_FOLDERS:
		if not os.path.exists(os.path.join(TEST_FOLDER, label)):
			os.makedirs(os.path.join(TEST_FOLDER, label))


LIST_KS = [10, 20, 40, 80, 160, 320, 640, 1280, 2560, 5120]
LIST_MS = [0, 10, 20, 40, 80, 160, 320, 640, 1280, 2560, 5120, 10240, 20480]

ITERATIONS = 10

results = {}

results['iterations'] = ITERATIONS
results['ks'] = LIST_KS
results['ms'] = LIST_MS

for i in range(1,(ITERATIONS+1)):
	print('# RUN ' + str(i) + ' of ' + str(ITERATIONS))
	runResult = {}
	for k in LIST_KS:
		kResult = {}
		for m in LIST_MS:
			print('# K: ' + str(k) + '| M: ' + str(m))
			# print('# M: ' + str(m))
			clearTrainFolders()
			fillTrainFolders(k,m,i)
			confusionMatrix = linear_classifier.runClassifier(TRAIN_FOLDER, TEST_FOLDER)
			kResult[m] = confusionMatrix
			print('-' * 80)
		runResult[k] = kResult
	results[i] = runResult
	
saveObject(results, 'experimentResults.pickle')