import os
import random
import time
from multiprocessing import Process, Manager, cpu_count
from multiprocessing.sharedctypes import RawArray, Value
import numpy as np
from numba import jit
import cuda
import sys




NUMBER_OF_COLUMNS = 10000
NUMBER_OF_ROWS = 10000
NUMBER_OF_PROCESS = 4

#Use GPU to calcuatate matrix addition
@jit
def MatrixAddtionOnGPU(matrix1, matrix2, matrix3):
    for i in range(NUMBER_OF_ROWS):
        for j in range(NUMBER_OF_COLUMNS):
            matrix3[i][j] = matrix1[i][j] + matrix2[i][j]
    return matrix3


#Simple CPU to calcuatate matrix addition
def MatrixAddtionOneCPUCore(matrix1, matrix2, matrix3):
    for i in range(NUMBER_OF_ROWS):
        for j in range(NUMBER_OF_COLUMNS):
            matrix3[i][j] = matrix1[i][j] + matrix2[i][j]
    return matrix3
   
    
def AddElement(startRow, matrix1, matrix2, matrix3):
    rowPerProccess = NUMBER_OF_ROWS // NUMBER_OF_PROCESS
    for x in range(rowPerProccess):
        for j in range(NUMBER_OF_COLUMNS):
           
            matrix3[startRow][j] = matrix1[startRow][j] + matrix2[startRow][j]
            
        startRow += 1    

def MatrixAddtionOnMultipleCPUCores(matrix1, matrix2, matrix3):
    rowPerProccess = NUMBER_OF_ROWS // NUMBER_OF_PROCESS
    startRow = 0
    proccesList = []
    
    while startRow < NUMBER_OF_ROWS:
        p = Process(target=AddElement, args=(startRow, matrix1, matrix2, matrix3))
        proccesList.append(p)
        matrix = p.start()
        startRow += rowPerProccess 
    
    for p in proccesList:
        p.join()
    
    return matrix3



def main():
    args = sys.argv
    debug = False

    
    #1 = norrmal, 2 =  multiple cpu cores, 3 = GPU
    typeOfAddtion = args[1]
    
    matrix1 = np.random.randint(1, 9, size=(NUMBER_OF_ROWS, NUMBER_OF_COLUMNS))
    matrix2 = np.random.randint(1, 9, size=(NUMBER_OF_ROWS, NUMBER_OF_COLUMNS))

    if typeOfAddtion == '1' or typeOfAddtion == '3':
        matrix3 = np.zeros(shape=(NUMBER_OF_ROWS, NUMBER_OF_COLUMNS))
    else:
        matrix3 = np.empty(dtype=object, shape=(NUMBER_OF_ROWS))
        matrix3.fill(RawArray("i", NUMBER_OF_COLUMNS))

    print("\nMatrices Size: %dx%d" % (NUMBER_OF_ROWS, NUMBER_OF_COLUMNS))

    #1 = norrmal, 2 =  multiple cpu cores, 3 = GPU
    if typeOfAddtion == '1':
        print("Single Core Used")
        print("------------------")
        start_time = time.time()
        newMatrix = MatrixAddtionOneCPUCore(matrix1, matrix2, matrix3)
    elif typeOfAddtion == '2':
        print("\nMultiple Cores Used")
        print("Number of Cores used " + str(NUMBER_OF_PROCESS))
        print("------------------")
        
        start_time = time.time()
        newMatrix = MatrixAddtionOnMultipleCPUCores(matrix1, matrix2, matrix3)
    else:
        print("\nGPU Used")
        print("------------------")
        start_time = time.time()
        newMatrix = MatrixAddtionOnGPU(matrix1, matrix2, matrix3)
    
    print("Completed in:")
    print("%s seconds" % (time.time() - start_time))

    
    if debug:
        for i in range(NUMBER_OF_ROWS):
            for j in range(NUMBER_OF_ROWS):
                if(j % (NUMBER_OF_ROWS-1) == 0 and j!=0):
                    print(str(newMatrix[i][j]) + " ")
                else:
                    print(str(newMatrix[i][j]) + " ", end='')

   

if __name__ == '__main__':
    main()