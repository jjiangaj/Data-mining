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
#assign3.py TRAIN TEST eps eta 
script = sys.argv[0]
train_file = sys.argv[1]
test_file = sys.argv[2]
eps = float(sys.argv[3])
eta = float(sys.argv[4])

"""
os.chdir(os.path.dirname(__file__)) 
train_file = "train.txt"
test_file = "test.txt"
eps = 0.001
eta = 0.0009
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

a0 = np.ones((test_datasize, 1))
test_aug = np.append(a0, test_data, axis = 1)
test_aug_attrisize = test_aug.shape[1]
test_aug = np.delete(test_aug, test_aug_attrisize-1, 1)  
test_aug_datasize = test_aug.shape[0]
test_aug_attrisize = test_aug.shape[1]
#print(test_aug_attrisize)

train_y = train_data[:,train_aug_attrisize - 1]
test_y = test_data[:,test_aug_attrisize - 1]
#print(test_y)

#--------------------------------Function definition: Sigmoid--------------------------------#
def sigmoid(x):
  return 1 / (1 + np.exp(-x))

#--------------------------------Logistic Regression: SGA--------------------------------#
t = 0 #iteration counter
w_t_1 = np.zeros(train_aug_attrisize)
w_t = np.zeros(train_aug_attrisize)
error = w_t_1 - w_t

while True:
    w = w_t
    for i in np.random.permutation(train_aug_datasize):
        gradient = (train_y[i] - sigmoid(np.dot(w,train_aug[i])))*train_aug[i]
        #print(gradient)
        w = w + eta*gradient
    w_t_1 = w_t
    w_t = w
    error = LA.norm(w_t - w_t_1)
    t = t+1
    #print(error) 
    #print(t)
    if(error < eps):
        break

count_correct = 0
for i in range(test_aug_datasize):
  y = sigmoid(np.dot(w_t, test_aug[i]))
  if (y>=0.5):
    y = 1
  else:
    y = 0
  if (y == test_y[i]):
    count_correct = count_correct+1

acc = count_correct/test_datasize

print('w value:', w_t)
print('Iteration =', t)
print('ACC = ',acc)