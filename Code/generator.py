import os

ks = [10, 20, 40, 80, 160, 320, 640, 1280, 2560, 5120]
ds = [6, 8]
iterations = 10


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

