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
filename = sys.argv[1]
threshold = float(sys.argv[2])

#read in data
input_data = np.loadtxt(filename, dtype=float, delimiter='\t')
datasize = input_data.shape[0]
attribute_size = input_data.shape[1]

#--------------------------------Part1.a--------------------------------#
print("Part1.a:")

mean = np.sum(input_data, axis=0)/datasize
print("Mean vector:\n", mean)

mean_matrix = np.tile(mean,(datasize,1))
input_centered = input_data + np.negative(mean_matrix)
var = np.sum(np.multiply(input_centered,input_centered),axis=0)/datasize
var = np.sum(var, axis=0)
print("Total Variance:\n", var)

#--------------------------------Part1.b--------------------------------#
#checking the variance:
#print("real cov:",np.cov(input_data, rowvar=False, bias=datasize))
#-----------------------------------------------------------------------#
print("Part1.b:")

inner_cov = np.matmul(np.transpose(input_centered), input_centered)/datasize
print("Covariance using Inner Product:\n", inner_cov)

outer_cov = np.zeros((attribute_size, attribute_size))
for i in range (datasize):
    point= input_centered[i]
    outer_cov = np.add(outer_cov,np.outer(np.transpose(point), point))
outer_cov = outer_cov/datasize
print("Covariance using Outer Product:\n", outer_cov)

#--------------------------------Part1.c--------------------------------#
print("Part1.c:")

#define a function for doing normalization
def normalize(input_vector):
    output_vector = input_vector/np.sqrt(np.dot(input_vector,input_vector))
    return output_vector

input_centered_transpose = np.transpose(input_centered)
input_attribute_normalized = normalize(input_centered_transpose[0]) #Take the first attribute and normalize it

for i in range (attribute_size - 1):
    new_attribute = normalize(input_centered_transpose[i+1]) #Normalize the following attributes
    input_attribute_normalized = np.vstack([input_attribute_normalized, new_attribute])

correlation_matrix = np.dot(input_attribute_normalized,np.transpose(input_attribute_normalized)) #Calculate the correlation matrix
print("The correlation matrix:\n", correlation_matrix)

# print(np.absolute(correlation_matrix).min()) #Find least correlated : -0.00366064

# np.fill_diagonal(correlation_matrix, 0) #Set diagonal as zeros so max & min can be found more easily
# print(correlation_matrix.max()) #Find most correlated : 0.75339378
# print(correlation_matrix.min()) #Find most anti-correlated : -0.50486815

plt.title('Most correlated', fontsize=20)
plt.xlabel('Angle of attack (degrees)')
plt.ylabel('Suction side displacement thickness (meters)')
plt.scatter(input_data[:,1],input_data[:,4],alpha = 0.6, s=10)
plt.show()

plt.title('Most anti-correlated', fontsize=20)
plt.xlabel('Angle of attack (degrees)')
plt.ylabel('Chord length (meters)')
plt.scatter(input_data[:,1],input_data[:,2],alpha = 0.6, s=10)
plt.show()

plt.title('Least correlated', fontsize=20)
plt.xlabel('Frequency (Hertz)')
plt.ylabel('Chord length (meters)')
plt.scatter(input_data[:,0],input_data[:,2],alpha = 0.6, s=10)
plt.show()

#--------------------------------Part2--------------------------------#

print("Part2:")

# orthogonalize the 2nd column of a dx2 matrix with respect to the first column
def orthogonalize(input_vector):

    # Check the input, make sure it's a dx2 matrix
    if(input_vector.shape[1]!=2):
        print("The input matrix for orthogonalization is not a dx2 matrix.")

    col_a = np.transpose(input_vector[:,0])
    col_b = np.transpose(input_vector[:,1])

    # The parameter is set negative to count for the minus
    para = -(np.dot( col_b, np.transpose(col_a)))/(np.dot(col_a,col_a))
    col_b = np.add(col_b, para * col_a)

    output_vector = np.transpose(np.vstack([col_a, col_b]))
    return output_vector

def normalize_d2(input_vector):
    # Check the input, make sure it's a dx2 matrix
    if(input_vector.shape[1]!=2):
        print("The input matrix for orthogonalization is not a dx2 matrix.")

    col_a = np.transpose(input_vector[:,0])
    col_b = np.transpose(input_vector[:,1])

    col_a = normalize(col_a)
    col_b = normalize(col_b)
    
    output_vector = np.transpose(np.vstack([col_a, col_b]))
    return output_vector

cov = inner_cov # Give this matrix a simpler name

# create two random vectors a0 and b0
a0 = np.ones(attribute_size)
b0 = np.arange(attribute_size)

# create the dx2 matrix
xn_0 = np.transpose(np.vstack([a0,b0]))

# this is the new matrix after multiplication
xn_1 = np.transpose(np.vstack([a0,b0]))

#threshold = float(input("Enter the threshold:")) # Set the ε
error = threshold

while True: # This is like do-while in C++ 

    # Calculate xn_1 from xn_0, then orthogonalize xn_1
    xn_1 = np.dot(cov, xn_0) 
    xn_1 = orthogonalize(xn_1)

    # Normalize each column in xn_0 and xn_1 to calculate error
    norm_xn_0 = normalize_d2(xn_0)
    norm_xn_1 = normalize_d2(xn_1)
    error = LA.norm( np.add(norm_xn_1, np.negative(norm_xn_0)))

    # update
    xn_0 = xn_1

    # terminating condition
    if(error < threshold):  
        break  

print("The eigen vectors found are (in columns):\n", normalize_d2(xn_0))

normalized_xn = normalize_d2(xn_0)
eig_value1 = np.dot(np.dot(normalized_xn[:,0], cov),np.transpose(normalized_xn[:,0]))
eig_value2 = np.dot(np.dot(normalized_xn[:,1], cov),np.transpose(normalized_xn[:,1]))
print("The corresponding eigen values found are:\n", eig_value1, eig_value2)

"""
# for checking output
print(error)
print(normalize_d2(xn_0))
print(normalize_d2(((np.dot(cov,xn_0))))
"""

proj = np.dot(input_centered, normalized_xn)


plt.title('Map onto eigenvector u1, u2', fontsize=20)
plt.xlabel('u1')
plt.ylabel('u2')
plt.scatter(proj[:,0],proj[:,1],alpha = 0.6, s=10)
plt.show()






