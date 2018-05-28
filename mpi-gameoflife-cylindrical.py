import random
import sys
import math
import time
import copy
from mpi4py import MPI
import numpy
import itertools
from matplotlib import pyplot as plt

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
stat = MPI.Status()

# Initialize Grid and Generation
prob = 0.7
COLS = 400
ROWS_TOTAL = 198
generations = 20
ROWS = ROWS_TOTAL//size+2


#==== init first array
N=numpy.random.binomial(1,prob,size=(ROWS+2)*COLS)
M=numpy.reshape(N,(ROWS+2,COLS))

M[0,:] = 0
M[ROWS+1,:] = 0
M[:,0] = 0
M[:,COLS-1] = 0

initM = numpy.copy(M)
print(initM)
print("First Generation")

#plt.imshow(initM, interpolation='nearest')
#plt.show()

#==== init first array

if size > ROWS:
    print("Not enough ROWS")
    exit()

# MPI core functions
def msgUp(subGrid):
	# Sends and Recvs rows with (Rank+1)%size for cylindricaly
    comm.send(subGrid[ROWS-2],dest=(rank+1)%size)
    subGrid[ROWS-1]=comm.recv(source=(rank+1)%size)
    return 0

def msgDn(subGrid):
	# Sends and Recvs rows with (Rank-1)%size for cylindrical
    comm.send(subGrid[1],dest=(rank-1)%size)
    subGrid[0] = comm.recv(source=(rank-1)%size)
    return 0

def computeGridPoints(M):
    intermediateM = numpy.copy(M)
    for ROWelem in range(1,ROWS+1):
        for COLelem in range(1,COLS-1):
            sum = ( M[ROWelem-1,COLelem-1]+M[ROWelem-1,COLelem]+M[ROWelem-1,COLelem+1]
                    +M[ROWelem,COLelem-1]+M[ROWelem,COLelem+1]
                    +M[ROWelem+1,COLelem-1]+M[ROWelem+1,COLelem]+M[ROWelem+1,COLelem+1] )
#               print(ROWelem," ",COLelem," ",sum)
            if M[ROWelem,COLelem] == 1:
                if sum < 2:
                        intermediateM[ROWelem,COLelem] = 0
                elif sum > 3:
                        intermediateM[ROWelem,COLelem] = 0
                else:
                        intermediateM[ROWelem,COLelem] = 1
                if M[ROWelem,COLelem] == 0:
                        if sum == 3:
                                intermediateM[ROWelem,COLelem] = 1
                        else:
                                intermediateM[ROWelem,COLelem] = 0
    M = numpy.copy(intermediateM)
    return M    



def printf(format, *args):
    sys.stdout.write(format % args)
    
def showgraph(nx,ny,arr):
    for i in range(len(arr)):
        for j in range(nx-1):
            if arr[i][j] == 1:
                printf('\033[45m'"%d "'\033[0m',arr[i][j])
            else:
                printf('\033[43m'"%d "'\033[0m',arr[i][j])
        printf("\n")
    printf('\x1b[2J\x1b[H')
    time.sleep(1)


for step in range(generations):
    # Game of Life Role
    computeGridPoints(M)
    # cylindrical boundaries : no need (rank==0) and (rank==n-1)	conditions
    msgUp(M)
    msgDn(M)
    tempGrid=comm.gather(M[1:ROWS-1],root=0)
    
    if rank == 0:
        newGrid=list(itertools.chain.from_iterable(tempGrid))
        if step%5 ==0: 
            print("-----------Generation:", step, "---------------")
            showgraph(COLS, ROWS, newGrid)
        