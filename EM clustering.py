import sys
import matplotlib.pyplot as plt
import numpy as np
import os
import csv
from numpy import linalg as LA
import math


""" Below are what's orginally used to implement the input
#CMD doesn't automatically change working dir to script's location
#So I do it manually
#os.chdir(os.path.dirname(__file__)) 
#filename = input("Enter the file name:")
"""
#assign6.py FILE k eps
# os.chdir(os.path.dirname(__file__)) 
# script = sys.argv[0]
# train_file = sys.argv[1]
# test_file = sys.argv[2]
# c = float(sys.argv[3])
# eps = float(sys.argv[4])
# ktype = sys.argv[5]
# spread = float(sys.argv[6])

# os.chdir(os.path.dirname(__file__)) 
# file = "dancing_truth.txt"
# k = 5
# eps = 0.001
# itr = 0

#assign6.py FILE k eps
os.chdir(os.path.dirname(__file__)) 
script = sys.argv[0]
file = sys.argv[1]
k = int(sys.argv[2])
eps = float(sys.argv[3])
itr = 0

data = []
#read in data
with open(file, newline='') as csvfile:
  reader = csv.reader(csvfile, delimiter = ',')

  for row in reader:
    if any(row):                 # Pick up the non-blank row of list
        data.append(row)              # Just for verification

data = np.array(data)
n = int(data.shape[0])
d = int(data.shape[1])

if(True):
  type = []
  for i in range(n):
    if data[i][d - 1] not in type:
      type.append(data[i][d - 1])

  type = np.asarray(type)

  def index(string):
    for i in range(type.shape[0]):
      if(string == type[i]): return i

  for i in range(n):
    data[i][d - 1] = index(data[i][d - 1])

y = data[:,d - 1]
y = y.astype(np.int)
data = np.delete(data, d-1, 1)  
data = data.astype(np.float)

#--------------------------------support---------------------------------------#
def norm_pdf_multivariate(x, mu, sigma):
    size = len(x)
    if x.shape == mu.shape and (size, size) == sigma.shape:
        sigma = sigma + 0.0001*np.identity(d-1)
        det = LA.det(sigma)
        if det == 0:
            raise NameError("The covariance matrix can't be singular")

        norm_const = 1.0/ ( math.pow((2*np.pi),float(size)/2) * math.pow(det,1.0/2) )
        x_mu = np.subtract(x, mu)
        inv = LA.inv(sigma)      
        result = math.pow(math.e, -0.5 * (np.dot(np.dot(x_mu,inv) ,x_mu.T)))
        return norm_const * result
    else:
        raise NameError("The dimensions of the input don't match")

def sum_p(j):
  sum = 0
  for a in range(k):
    sum = sum + norm_pdf_multivariate(data[j],miu[a],sig[a])*prior[i]
  # print('sum:',sum, j)
  return sum
#--------------------------------Initialization--------------------------------#
miu = np.random.rand(k,d-1)
sig = np.empty([k,d-1,d-1])
prior = np.full((k),1/k)

for i in range(k):
  sig[i] = np.identity(d-1)

#--------------------------------EM clustering---------------------------------#
conf = np.zeros([k,k])
cluster = np.zeros(k)
y_hat = np.empty(n)

def EM_test(w):
  for j in range(n):
    index = 0
    value = 0
    y_hat[j] = index = np.argmax(w[:,j])
    t = y[j]
    conf[index][t] += 1
    cluster[index] += 1
  return
#--------------------------------EM clustering---------------------------------#
err = 0
while True:
  itr += 1
  w = np.empty([k,n])
  for i in range(k):
    for j in range(n):
      w[i][j] = norm_pdf_multivariate(data[j],miu[i],sig[i])*prior[i]/sum_p(j)
    
  old_miu = np.copy(miu)
  for i in range(k):
    miu[i] = np.dot(data.T, w[i])/np.dot(w[i], np.ones(n))
    sum_sig = 0
    for j in range(n):
      z = data[j] - miu[i]
      sum_sig = sum_sig + w[i][j] * np.outer(z, z.T)
    sig[i] = sum_sig/np.dot(w[i], np.ones(n))
    prior[i] = np.dot(w[i], np.ones(n))/n
    
  err = 0
  for i in range(k):
      err = err + LA.norm(old_miu[i] - miu[i])

  if(err < eps):
    break

EM_test(w)

purity = 0
for i in range(k):
  purity += np.amax(conf[i])

purity /= n

print("Final mean for each cluster:\n", miu)
print("Final covariance for each cluster:\n", sig)
print("# of iterations:\n", itr)
print("Cluster assignment, confusion matrix:\n", conf)
print("size of each cluster", cluster)
print("Purity", purity)