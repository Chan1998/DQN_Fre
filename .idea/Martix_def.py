import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns

#设定超参数
M = 1               #可用基站数
K = 3              #可用基站频点
N = 7          #申请用户数量

def Location_matrix_df(n,m,k):
    Location_matrix = np.zeros(shape=(n,m,k),dtype=int)
    for i in range (n):
        Location_matrix[i,int(m * random.random()),0:k] = 1
    return Location_matrix

def Location_matrix_show(Location_matrix):
    Location_matrix_2D = Location_matrix[:,:,0]
    sns.heatmap(Location_matrix_2D ,annot=False, vmin=0, vmax=2, cmap="Blues", xticklabels=False  ,
                    yticklabels =False   )
    plt.xlabel ('Base')
    plt.ylabel ('Users')
    plt.show()

def Random_Allocation_matrix_df(n,m,k,Location_matrix):
    Random_Allocation_matrix = np.zeros(shape=(n,m,k),dtype=int)
    for i in range (n):
        #for j in range(m):
            #if np.all(Location_matrix[i,j,0])==1:

                if  np.all(Random_Allocation_matrix[i,0,:])==1:
                    print("User %d don't have free Frequence"  % (i))

                else:
                    for l in range (k):
                        if np.all(Random_Allocation_matrix[:,0,l])==0:
                            Random_Allocation_matrix[i,0,l] = 1
                            break

    return Random_Allocation_matrix


Location_matrix = Location_matrix_df(N, M, K)
#print(Location_matrix)
#Location_matrix_show(Location_matrix)

Random_Allocation_matrix = Random_Allocation_matrix_df(N, M, K,Location_matrix)

print(Random_Allocation_matrix)