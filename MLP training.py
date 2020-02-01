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
"""
#assign3.py TRAIN TEST eps eta 
script = sys.argv[0]
train_file = sys.argv[1]
test_file = sys.argv[2]
eps = float(sys.argv[3])
eta = float(sys.argv[4])
"""
#assign4.py TRAIN TEST m η epochs
os.chdir(os.path.dirname(__file__)) 
script = sys.argv[0]
train_file = sys.argv[1]
test_file = sys.argv[2]
m = int(sys.argv[3])
eta = float(sys.argv[4])
epochs = float(sys.argv[5])

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

p = int(np.max(train_y))
d = int(train_attrisize)
n = int(train_datasize)

#--------------------------------Function definition: Sigmoid, ReLU, Softmax--------------------------------#
def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def ReLU(x):
  if(x > 0):
    return x
  else:
    return 0

def ReLu_derr(array):
  derivative = np.zeros(len(array))
  for i in range(len(array)): 
    if(array[i] > 0):
      derivative[i] = 1
    else:
      derivative[i] = 0
  return derivative

def Softmax(array):
  exp = np.zeros(len(array))
  softmax = np.zeros(len(array))
  sum = 0
  for i in range(len(array)):
    exp[i] = np.exp(array[i])
    sum += exp[i]
  for i in range(len(array)):
    softmax[i] = exp[i]/sum
  return softmax

#--------------------------------Function definition: One-hot--------------------------------#
def one_hot(train):
  code = np.identity(p)
  encoded = []
  for i in train:
    encoded = np.append(encoded, code[int(i-1)])
  encoded = np.reshape(encoded,(n,p))
  return encoded

#-----------------------------------Initialization-----------------------------------#
onehot_train = one_hot(train_y)
bh = np.random.randint(100,size = m)/16000
bo = np.random.randint(100,size = p)/16000
Wh = np.random.randint(100,size = (d,m))/16000
Wo = np.random.randint(100,size = (m,p))/16000

t = 0
while True:
  z = np.zeros(m)
  o = np.zeros(p)
  neth = np.zeros(m)
  neto = np.zeros(p)
  onesh = np.ones(m)
  oneso = np.ones(p)

  for i in np.random.permutation(n):
    #-----------------------------------Forward prop-----------------------------------#
    neth = bh + np.dot(Wh.T, train_data[i])
    for j in range(m):
      z[j] = ReLU(neth[j])
    neto = bo + np.dot(np.transpose(Wo), z)
    o = Softmax(neto)
    #-----------------------------------Back prop-----------------------------------#
    erroro = o - onehot_train[i]
    deltao =  erroro
    deltah = ReLu_derr(z) * np.dot(Wo,deltao)
    #-----------------------------------Gradient descent bias-----------------------------------#
    gradient_bo = deltao
    gradient_bh = deltah
    bo = bo - eta*gradient_bo
    bh = bh - eta*gradient_bh
    #-----------------------------------Gradient descent weight-----------------------------------#
    gradient_Wo = np.outer(z,deltao.T)
    gradient_Wh = np.outer(train_data[i],deltah.T)
    Wo = Wo - eta*gradient_Wo
    Wh = Wh - eta*gradient_Wh

  t += 1
  if(t>epochs): 
    break
#-----------------------------------Testing-----------------------------------#
count = 0
for i in range(test_datasize):
  z = np.zeros(m)
  o = np.zeros(p)
  neth = bh + np.dot(Wh.T, test_data[i])
  for j in range(m):
    z[j] = ReLU(neth[j])
  neto = bo + np.dot(Wo.T, z)
  o = Softmax(neto)
  y_hat = np.argmax(o) + 1
  if(y_hat == test_y[i]): 
    count = count+1

ACC = count/test_datasize

print("bh:",bh)
print("bo:",bo)
print("Wh:",Wh)
print("Wo:",Wo)
#print("o:",o)
print(ACC*100)

