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

#assign3-kernel.py TRAIN TEST [linear | quadratic | gaussian] [spread]
script = sys.argv[0]
train_file = sys.argv[1]
test_file = sys.argv[2]
type = sys.argv[3]
spread = float(sys.argv[4])
alpha = 0.01

"""
os.chdir(os.path.dirname(__file__)) 
train_file = "train.txt"
test_file = "test.txt"
type = "gaussian"
spread = 1.5
alpha = 0.01
"""

#read in data
train_data = np.loadtxt(train_file, dtype=float, delimiter=',')
test_data = np.loadtxt(test_file, dtype=float, delimiter=',')
train_datasize = train_data.shape[0]
train_attrisize = train_data.shape[1]
test_datasize = test_data.shape[0]
test_attrisize = test_data.shape[1]

#--------------------------------Create augmented matrix--------------------------------#
train_y = train_data[:,train_attrisize - 1]
test_y = test_data[:,test_attrisize - 1]

train_data = np.delete(train_data, train_attrisize-1, 1)  
train_datasize = train_data.shape[0]
train_attrisize = train_data.shape[1]

test_data = np.delete(test_data, test_attrisize-1, 1)  
test_datasize = test_data.shape[0]
test_attrisize = test_data.shape[1]

#print(train_attrisize)
#print(train_y)

#--------------------------------Function definitions--------------------------------#
def kernel_linear(kernel_1, kernel_2, size1, size2):
    for i in range(size1):
        for j in range(size2):
            kernel[i][j] = 1 + np.dot(kernel_1[i],kernel_2[j])
    return kernel

def kernel_quadratic(kernel_1, kernel_2, size1, size2):
    for i in range(size1):
        for j in range(size2):
            kernel[i][j] = 1 + np.square(np.dot(kernel_1[i],kernel_2[j]))
    return kernel

def kernel_gaussian(kernel_1, kernel_2, size1, size2):
    for i in range(size1):
        for j in range(size2):
            diff = kernel_1[i] - kernel_2[j]
            exponent = - np.dot(diff, diff)/(2*spread*spread)
            kernel[i][j] = 1 + np.exp(exponent)
    return kernel


#--------------------------------Calculate Kernel matrix--------------------------------#
kernel = np.zeros((train_datasize, train_datasize))
if(type == "linear"):
  kernel = kernel_linear(train_data, train_data, train_datasize, train_datasize)
  #print(kernel)
elif(type == "quadratic"):
  kernel = kernel_quadratic(train_data, train_data, train_datasize, train_datasize)
elif(type == "gaussian"):
  kernel = kernel_gaussian(train_data, train_data, train_datasize, train_datasize)

identity = np.identity(train_datasize)
inverse = np.linalg.inv(kernel + alpha*identity)

c = np.dot(inverse, train_y)

#print(c)

#--------------------------------Test Kernel matrix--------------------------------#
y_hat = np.zeros(test_datasize)
kernel_z = []

if(type == "linear"):
  kernel_z = kernel_linear(test_data, train_data, test_datasize, train_datasize)
  #print(kernel)
elif(type == "quadratic"):
  kernel_z = kernel_quadratic(test_data, train_data, test_datasize, train_datasize)
elif(type == "gaussian"):
  kernel_z = kernel_gaussian(test_data, train_data, test_datasize, train_datasize)

for i in range(test_datasize):
    for j in range(train_datasize):
       y_hat[i] = y_hat[i] + c[j]*kernel_z[i][j]


count_correct = 0
for i in range(test_datasize):
  y = 0
  if (y_hat[i]>=0.5):
    y = 1
  else:
    y = 0
  if (y == test_y[i]):
    count_correct = count_correct+1
if(type == "gaussian"):
  print("Using", type,"kernel, with spread:", spread)
else:
  print("Using", type,"kernel")
  
print("ACC:", count_correct/test_datasize)
