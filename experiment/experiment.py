import os
import shutil
import linear_classifier
import pickle


TRAIN_FOLDER = 'training'
TEST_FOLDER = 'test'
LABEL_FOLDERS = ['6', '8']

LIST_KS = [100, 500, 1000, 5000]
LIST_MS = [0, 100, 500, 1000, 5000, 10000]

RERUNS_PER_K_M_COMBI = 10

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

def fillTrainFolders(numberOfOriginal, numberOfGenerated, startWith=0):
	# print('### Filling training folders with each ' + str(numberOfOriginal) + ' originals and ' 
		# + str(numberOfGenerated) + ' generated (starting at generated_' + str(startWith) + '.png)')
	for label in LABEL_FOLDERS:
		folder = label + '_' + str(numberOfOriginal)
		folder = os.path.join(folder, 'samples')
		# copy original images:
		for imgId in range(0, numberOfOriginal):
			fileName = 'original_' + str(imgId) + '.png'
			filePath = os.path.join(folder, fileName)
			shutil.copy2(filePath, os.path.join(TRAIN_FOLDER, label))
		# copy generated images:
		for imgId in range(0, numberOfGenerated):
			fileName = 'generated_' + str(imgId + startWith) + '.png'
			filePath = os.path.join(folder, fileName)
			shutil.copy2(filePath, os.path.join(TRAIN_FOLDER, label))
	return

def saveObject(obj, filename):
    with open(filename, 'wb') as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

results = {}
results['iterations'] = RERUNS_PER_K_M_COMBI
results['ks'] = LIST_KS
results['ms'] = LIST_MS

for i in range(0,RERUNS_PER_K_M_COMBI):
	print('# RUN ' + str(i+1) + ' of ' + str(RERUNS_PER_K_M_COMBI))
	runResult = {}
	for k in LIST_KS:
		kResult = {}
		for m in LIST_MS:
			print('# K: ' + str(k) + '| M: ' + str(m))
			# print('# M: ' + str(m))
			clearTrainFolders()
			fillTrainFolders(k,m,i*m)
			confusionMatrix = linear_classifier.runClassifier(TRAIN_FOLDER, TEST_FOLDER)
			kResult[m] = confusionMatrix
			print('-' * 80)
		runResult[k] = kResult
	results[i] = runResult
	
saveObject(results, 'experimentResults.pickle')

# with open('experimentResults.pickle', 'rb') as input:
# 	results = pickle.load(input)
# 	# print number of iterations:
# 	print(results['iterations'])
# 	# print list of used ks (number of original samples):
# 	print(results['ks'])
# 	# print list of used ms (number of generated samples):
# 	print(results['ms'])
# 	# first iteration, 100 original, 0 generated:
# 	print(results[0][100][0])


	




