import os

# Define the parameters of the generator:
# Train networks based on k number of orignals:
ks = [10, 20, 40, 80, 160, 320, 640, 1280, 2560, 5120]
# Train for the following digits:
ds = [6, 8]
# Train this amount of different networks for each combination of k and d:
iterations = 10


# Construct the python command and let the OS execute it:
for i in range(1, (iterations+1)):
	for k in ks:
		for d in ds:
			print('### Iteration ' + str(i) + ' of ' + str(iterations))
			print('### k: ' + str(k))
			print('### d: ' + str(d))
			subDir = str(d) + '_' + str(k) + '_' + str(i)
			command = 'python main.py --dataset mnist --is_train True'
			command = command + ' --sample_dir samples/' + subDir
			command = command + ' --checkpoint_dir checkpoint/' + subDir
			command = command + ' --d ' + str(d)
			command = command + ' --k ' + str(k)
			command = command + ' > ' + subDir +'.txt'
			os.system(command)

