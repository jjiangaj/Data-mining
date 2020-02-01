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

#assign5.py Attach:iris-virginica.txt 1 0.001 linear
#assign5.py Attach:iris-versicolor.txt 1 0.001 linear
#assign5.py Attach:iris-versicolor.txt 1 0.001 quadratic
#assign5.py TRAIN TEST C eps [linear OR quadratic OR gaussian ] spread
os.chdir(os.path.dirname(__file__)) 
script = sys.argv[0]
train_file = sys.argv[1]
test_file = sys.argv[2]
c = float(sys.argv[3])
eps = float(sys.argv[4])
ktype = sys.argv[5]
spread = float(sys.argv[6])

# os.chdir(os.path.dirname(__file__)) 
# train_file = "train.txt"
# test_file = "test.txt"
# eps = 0.001
# c = 10
# spread = 0.5
# ktype = "gaussian"

#read in data
train_data = np.loadtxt(train_file, dtype=float, delimiter=',')
test_data = np.loadtxt(test_file, dtype=float, delimiter=',')
train_attrisize = train_data.shape[1]
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

d = int(train_attrisize)
n = int(train_datasize)

#--------------------------------support--------------------------------#

def kernel( kernel_1, kernel_2, type):

  size1 = kernel_1.shape[0]
  size2 = kernel_2.shape[0]
  kernel = np.zeros((size1, size2))

  if(type == "linear"):
    for i in range(size1):
        for j in range(size2):
            kernel[i][j] = 1 + np.dot(kernel_1[i],kernel_2[j])
    return kernel
  elif(type == "quadratic"):
    for i in range(size1):
        for j in range(size2):
            kernel[i][j] = 1 + np.square(np.dot(kernel_1[i],kernel_2[j]))
    return kernel
  elif(type == "gaussian"):
    for i in range(size1):
        for j in range(size2):
            diff = kernel_1[i] - kernel_2[j]
            exponent = - np.dot(diff, diff)/(2*spread*spread)
            kernel[i][j] = 1 + np.exp(exponent)
    return kernel

#--------------------------------SVM--------------------------------#
def SVM( c, type):
  kernelM = kernel(train_data, train_data, type)
  eta = np.zeros(n)
  for k in range(n):
    eta[k] = 1/kernelM[k][k]

  # print(eta)
  t = 0
  alpha_prev = np.zeros(n)
  alpha_next = np.zeros(n)

  while True:
    alpha_temp = np.copy(alpha_next)
    for k in range(n):
      sum = 0
      # for i in range(n):
      #   if(alpha_temp[k] == 0):break
      #   sum += alpha_temp[i]*train_y[i]*kernelM[i][k]
      sum = np.multiply(np.multiply( alpha_temp, train_y), kernelM[:,k])
      sum = np.sum(sum)
      alpha_temp[k] += eta[k]*(1 - train_y[k]*sum )
      if(alpha_temp[k] < 0 ): alpha_temp[k] = 0
      if(alpha_temp[k] > c ): alpha_temp[k] = c
    alpha_prev = np.copy(alpha_next)
    alpha_next = np.copy(alpha_temp)
    t += 1
    err = LA.norm(alpha_prev - alpha_next)
    if(err <= eps):
      break
  return alpha_next

def testSVM( alpha, data, type):
  kernelM = kernel(train_data, data, type)
  y_hat = np.zeros(data.shape[0])
  for k in range(data.shape[0]):
    sum = 0
    for i in range(n):
      sum += alpha[i]*train_y[i]*kernelM[i][k]
    y_hat[k] = np.sign(sum)
  return y_hat

alpha = SVM(c, ktype)
y = testSVM( alpha, test_data, ktype)
count = 0
for i in range(test_datasize):
  if(y[i] == test_y[i]):
    count+=1

acc = count/test_datasize
print("Accuracy rate = ", acc*100, "%")
for i in range(n):
  if(alpha[i] != 0):
    print("( ",i,", ",alpha[i],")")


if(ktype == "linear" ):
# for linear phi(x) -> x
# x is augmented so the last element is for bias
# w = sum(alpha*yi*xi)

  a0 = np.ones((n, 1))
  train_aug = np.append(train_data, a0, axis = 1)
  w = np.zeros(d + 1)

  for i in range(n):
    w = w + alpha[i]*train_y[i]*train_aug[i]
  print("w is: ", w)

# for quadratic phi(x) ->  (x1^2, x2^2, x3^2, x4^2, x5^2, x6^2, x7^2, x8^2, 
#                     sqrt2(x1x2, x1x3, x1x4, x1x5, x1x6, x1x7, x1x8), # meaning that each term is multiplied by sqrt(2)
#                     sqrt2(x2x3, x2x4, x2x5, x2x6, x2x7, x2x8),
#                     sqrt2(x3x4, x3x5, x3x6, x3x7, x3x8),
#                     sqrt2(x4x5, x4x6, x4x7, x4x8),
#                     sqrt2(x5x6, x5x7, x5x8),
#                     sqrt2(x6x7, x6x8),
#                     sqrt2(x7x8)
# phi(x) is augmented so the last element is for bias
if(ktype == 'quadratic'):
  w = np.zeros(int(d*(d+1)/2)+1)
  phi = np.zeros((n, int(d*(d+1)/2)))
  for i in range(n):
    for j in range(d):
      phi[i, j] = train_data[i, j]*train_data[i, j]
    t = d
    for j in range(0, d):
      for k in range(j+1, d):
        phi[i, t] = np.sqrt(2)*train_data[i, j]*train_data[i, k]
        t = t+1
  a0 = np.ones((n, 1))
  phi_aug = np.append(phi, a0, axis = 1)
  for i in range(n):
    w = w + alpha[i]*train_y[i]*phi_aug[i]
  print("w is: ", w)

if(ktype == 'gaussian'):
  print("w is not required")
  
  