import numpy as np

def magic_d(weights,values,capacity):

    items = len(values);
    Mdim = np.zeros([capacity+1,2])

    # If this takes too long use dragon_ball from Namek; they are more powerfull.
    for dragon_ball in range(0,len(weights)):
        for cap in range(1,capacity+1):
            if weights[dragon_ball] <= cap:
                Mdim[cap,1] = max( Mdim[cap,0], values[dragon_ball] + Mdim[cap-weights[dragon_ball],0])
            else:
                Mdim[cap,1] = Mdim[cap,0]
        Mdim[:,0]=Mdim[:,1]
    value = Mdim[capacity,1]
    return value


def solveIt(inputData):
    # Modify this code to run your optimization algorithm

    # parse the input
    lines = inputData.split('\n')

    firstLine = lines[0].split()
    items = int(firstLine[0])
    capacity = int(firstLine[1])

    values = []
    weights = []
    
    for i in range(1, items+1):
        line = lines[i]
        parts = line.split()

        values.append(int(parts[0]))
        weights.append(int(parts[1]))

    items = len(values)

    # weights is a list containing the different weights for the items
    # values is a list containing the different values for the items


    # WRITE YOUR OWN CODE HERE #####################################
    
    ## MAGIC ##
    # best_value = magic_d(weights,values,capacity)

    value =0
    taken = items*[0]

    value = magic_d(weights,values,capacity)

    
    # STOP WRITING YOUR CODE HERE ###################################
    

    outputData = str(value) + ' ' + str(0) + '\n'
    outputData += ' '.join(map(str, taken))
    return outputData


import sys

if __name__ == '__main__':
    if len(sys.argv) > 1:
        fileLocation = sys.argv[1].strip()
        inputDataFile = open(fileLocation, 'r')
        inputData = ''.join(inputDataFile.readlines())
        inputDataFile.close()
        print solveIt(inputData)
    else:
        print 'This test requires an input file.  Please select one from  data_ninjas (i.e. python solver.py ./data/ninjas_1_4)'
        # EXAMPLE of execution from terminal: 
        #      python solver.py ./data/ninja_1_4
