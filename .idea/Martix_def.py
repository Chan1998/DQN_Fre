import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
#import math

#设定超参数
M = 4              #可用基站数
K = 2              #可用基站频点
N = 6               #申请用户数量

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
    flag = 0
    for i in range(n):
        for l in range(m):
            if np.all(Location_matrix[i, l, 0]) == 1:
                for x in range(k):
                    for j in range(n):
                        flag = flag + Random_Allocation_matrix[j, l, x]  # 观察x号频段是否有人使用

                    if flag == 1:
                        flag = 0
                        print("对于基站%d范围内%d号用户,频段 %d 已被占用" % (l,i,x))
                    else:
                        flag = 0
                        Random_Allocation_matrix[i, l, x] = 1
                        print("基站%d范围内%d号用户,频段 %d 成功分配" % (l, i, x))
                        print("%d号用户,随机分配成功" % ( i))
                        break
        if np.sum(Random_Allocation_matrix[i,:,:])==0:
            print("%d号用户,随机分配失败" % ( i))

    return Random_Allocation_matrix

def I_caculate(n,m,k,Allocation_matrix):
    I_matrix = np.zeros(shape=(n,m,k),dtype=int)
    for l in range (k):
        for i in range (n):
            for j in range (m):
                I_matrix[i,j,l] = np.sum(Allocation_matrix[:,:,l]) - np.sum(Allocation_matrix[:,j,l])
    return I_matrix

def R_caculate(n,m,k,Allocation_matrix,I_matrix):
    Allocation_matrix_float = Allocation_matrix.astype(np.float)
    r = np.sum(np.log2(1 + Allocation_matrix_float/I_matrix))
    return r

Location_matrix = Location_matrix_df(N, M, K)
#print(Location_matrix)
#Location_matrix_show(Location_matrix)

Random_Allocation_matrix = Random_Allocation_matrix_df(N, M, K,Location_matrix)

print(Random_Allocation_matrix)
Allocation_matrix = Random_Allocation_matrix
I_matrix = I_caculate(N, M, K,Allocation_matrix)
print(I_matrix)
r = R_caculate(N, M, K,Allocation_matrix,I_matrix)
print ("对于当前矩阵，总传输速率为%g"%(r))