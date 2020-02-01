import sys
import matplotlib.pyplot as plt
import numpy as np
import os
from numpy import linalg as LA

""" Below are what's orginally used to implement the input
#CMD doesn't automatically change working dir to script's location
#So I do it manually
#os.chdir(os.path.dirname(__file__)) 
#filename = input("Enter the file name:")
"""
#dealing with the input
script = sys.argv[0]
train_file = sys.argv[1]
test_file = sys.argv[2]
alpha = float(sys.argv[3])

"""
os.chdir(os.path.dirname(__file__)) 
train_file = "wanshouxigong.txt"
test_file = "Wanliu.txt"
alpha = 0.0001
"""

#read in data
train_data = np.loadtxt(train_file, dtype=float, delimiter=',')
test_data = np.loadtxt(test_file, dtype=float, delimiter=',')
train_datasize = train_data.shape[0]
train_attrisize = train_data.shape[1]
test_datasize = test_data.shape[0]
test_attrisize = test_data.shape[1]

#--------------------------------Create augmented matrix--------------------------------#
a0 = np.ones((train_datasize, 1))
train_aug = np.append(a0, train_data, axis = 1)
train_aug_attrisize = train_aug.shape[1]
train_aug = np.delete(train_aug, train_aug_attrisize-1, 1)  
train_aug_datasize = train_aug.shape[0]
train_aug_attrisize = train_aug.shape[1]

test_aug = np.append(a0, test_data, axis = 1)
test_aug_attrisize = test_aug.shape[1]
test_aug = np.delete(test_aug, test_aug_attrisize-1, 1)  
test_aug_datasize = test_aug.shape[0]
test_aug_attrisize = test_aug.shape[1]
#print(test_aug_attrisize)

train_y = train_data[:,train_aug_attrisize - 1]
test_y = test_data[:,test_aug_attrisize - 1]

#--------------------------------Create modification for ridge regression--------------------------------#
alpha_mat = np.sqrt(alpha)*np.identity(train_aug_attrisize)
#print(alpha_mat)
train_aug_noridge = train_aug
train_aug = np.concatenate((train_aug, alpha_mat),axis=0)
train_aug_datasize = train_aug.shape[0]

train_y_noridge = train_y
train_y = np.concatenate((train_y,np.zeros(train_aug_attrisize)),axis=0)
#print(train_y.shape[0])
#print(train_aug.shape[1])

#--------------------------------Find QR Matrix--------------------------------#

train_q = []
train_r = np.zeros((train_aug_attrisize, train_aug_attrisize),float)
np.fill_diagonal(train_r,1)
#print(train_aug[:,1])

train_q = np.array(train_q)
train_r = np.array(train_r)

def proj(u, A):
    return np.dot(u, A)/np.dot(u, u)

def col(matrix, num):
    if(matrix.ndim == 1): return matrix[:]
    else: return matrix[:,num]

for i in range (train_aug_attrisize):
    u = train_aug[:,i]
    #print(u)
    #print(train_q)
    if i == 0: 
        train_q = u
        continue
    for j in range(i):
        #print(col(train_q, j))
        #print(train_aug[:,i])
        project = proj( col(train_q, j), train_aug[:,i])
        train_r[j][i] = project
        #print(project)
        u = np.add( u, (-project)*col(train_q, j))
    
    u = np.reshape(u,(train_aug_datasize,1))
    train_q = np.reshape(train_q,(train_aug_datasize,i))
    train_q = np.hstack((train_q, u))
    
#print(train_r)

#--------------------------------Compute inverse delta--------------------------------#
inverse_delta = np.zeros((train_aug_attrisize, train_aug_attrisize),float)
for i in range(train_aug_attrisize):
    inverse_delta[i,i] = 1/np.dot(train_q[:,i],train_q[:,i])
#print(train_q)
#print(train_r)
#print(inverse_delta)

#--------------------------------Find w by backsubstitution--------------------------------#
backsolve = np.matmul(inverse_delta, np.transpose(train_q))
backsolve = np.dot(backsolve, train_y)
#print(backsolve)

w = np.zeros((train_aug_attrisize,), float)
for i in range(train_aug_attrisize-1, -1, -1):
    w[i] = backsolve[i]
    #print("----",i,"---")   
    for j in range(train_aug_attrisize - 1, i, -1):
        w[i] = w[i] - w[j]*train_r[i][j]
        #print("w[",j,"]:",w[j],",coeff:",train_r[i][j])

print("The weight vector w is:\n",w)
print("The L2 norm of w is:\n",np.sqrt(np.dot(w,w)))

#--------------------------------Calculate SSE--------------------------------#
SSE = 0
SSE_test = 0
for i in range(train_datasize):
    error = np.dot(w, train_aug_noridge[i]) - train_y[i]
    SSE += np.square(error)
    
for i in range(test_aug_datasize):
    error_test = np.dot(w, test_aug[i]) - test_y[i]
    SSE_test += np.square(error_test)

print("The SSE of train dataset is:\n",SSE)
print("The SSE of test dataset is:\n",SSE_test)

#--------------------------------Calculate R^2--------------------------------#

TSS = 0
TSS_test = 0
mean = np.mean(train_y_noridge)
mean_test = np.mean(test_y)

for i in range(train_datasize):
    diff = train_y_noridge[i] - mean
    diff_test = test_y[i] - mean_test
    TSS += np.square(diff)
    TSS_test += np.square(diff_test)

Rsqr = ( TSS - SSE )/ TSS
Rsqr_test = ( TSS_test - SSE_test )/ TSS_test

print("The R^2 of train dataset is:\n",Rsqr)
print("The R^2 of test dataset is:\n",Rsqr_test)

#--------------------------------Find out change caused by an increase of 1 sdv --------------------------------#

stdev = []
for i in range(test_aug_attrisize):
    stdev = np.append(stdev, w[i] * np.std(test_aug[:,i]))

print("Change caused by 1 sdv in test set:")
print(stdev)