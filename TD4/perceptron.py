# -*- coding: utf-8 -*-
"""
**SI221 Assignment #4**

**1. Synthetic data**

### **Question 1**
"""

from random import *
import numpy as np 
import matplotlib.pyplot as plt


''' Function to generate a random dataset '''
def generate_data_set(sigma):
    y = np.zeros(200)
    v = sample(range(200), 100)
    mean = []
    cov = [[sigma,0],[0,sigma]]
    x =[]
    for m in v : 
        y[m] = 1
    for i in range(200):
        if (y[i] == 0):
            mean = [-1,0]
        elif (y[i] == 1):
            mean = [1,0]
        x.append(np.concatenate((np.random.multivariate_normal(mean, cov),[1]),axis=None))
    return x, y


''' Function that returns the prediciton given the weight vector and the input vector '''
def prediction(w,x):
    if np.dot(w,x) > 0 :
        return 1
    else :
        return 0


''' Implementation of the perceptron algorithm '''
def perceptron(x, y, eta):
    w = np.zeros(3)
    cpt=0
    while (cpt<200) :
        for i in range(200):
            if (y[i]-prediction(w,x[i]) == 1):
                w = w + eta*x[i]
            elif (y[i]-prediction(w,x[i]) == -1):
                w = w - eta*x[i]
        cpt+=1
    return w
        

''' Compute the average error and the standard deviation '''
def average_error_and_standard_deviation(sigma):

  average_error = 0
  standard_deviation = 0
  L = []

  # On prend eta=1
  for i in range (50):
    x,y = generate_data_set(sigma)
    w = perceptron(x, y, 1)
    s = sum([abs(y[i] - prediction(w,x[i])) for i in range(200)]) / 200
    L.append(s)
    average_error += s/50

  for i in range (50):
    standard_deviation += (L[i] - average_error)**2

  standard_deviation = (standard_deviation/50)**(1/2)

  return [average_error,standard_deviation]

#Plot error bars

abscisse = [0.05, 0.25, 0.50, 0.75]
labels = ['0.05', '0.25', '0.50', '0.75']
x_pos = np.arange(len(abscisse))
CTEs = [average_error_and_standard_deviation(i)[0] for i in abscisse]
error = [average_error_and_standard_deviation(i)[1] for i in abscisse]

fig, ax = plt.subplots()
ax.bar(x_pos, CTEs, yerr=error, align='center', alpha=0.5, ecolor='black', capsize=10)
ax.set_ylabel('Average error')
ax.set_xticks(x_pos)
ax.set_xticklabels(labels)
ax.set_title('Average error for different values of variance')
ax.yaxis.grid(True)

plt.tight_layout()
plt.show()

"""### **Question 2**"""

''' Function to find the new labels '''
def flip_y(y, p):
    for i in range(200):
        rd = random()
        if (rd < p):
            y[i] = 1 - y[i]
    return y


''' Compute the average error and standard deviation for the new dataset generated for every probability '''
def average_error_and_standard_deviation_proba(p):

  average_error = 0
  standard_deviation = 0
  L = []
  
  # On prend eta=1
  for i in range (50):
    X1, y1 = generate_data_set(0.15)
    y = flip_y(y1, p)
    w = perceptron(X1, y, 1)
    s = sum([abs(y[i] - prediction(w,X1[i])) for i in range(200)]) / 200
    L.append(s)
    average_error += s/50

  for i in range (50):
    standard_deviation += (L[i] - average_error)**2

  standard_deviation = (standard_deviation/50)**(1/2)

  return [average_error,standard_deviation]

#Plot error bars

abscisse = [0, 0.05, 0.1, 0.2]
labels = ['0', '0.05', '0.1', '0.2']
x_pos = np.arange(len(abscisse))
CTEs = [average_error_and_standard_deviation_proba(i)[0] for i in abscisse]
error = [average_error_and_standard_deviation_proba(i)[1] for i in abscisse]

fig, ax = plt.subplots()
ax.bar(x_pos, CTEs, yerr=error, align='center', alpha=0.5, ecolor='black', capsize=10)
ax.set_ylabel('Average error')
ax.set_xticks(x_pos)
ax.set_xticklabels(labels)
ax.set_title('Average error for different values of p')
ax.yaxis.grid(True)

plt.tight_layout()
plt.show()

"""### **Image data: LANDSAT on Tarascon**

## **Labeling data**
"""

import tiilab

img=tiilab.imz2mat("landsattarasconC4.ima") 
tiilab.visusar(img[0])

print(img[0])

''' Function to create labels from an image '''
def create_labels(img):
    image = img[0]
    line  = len(image)
    row   = len(image[0])

    label_matrix = np.zeros((line,row))

    for i in range(line):
      for j in range(row):
        if image[i][j]<30:
          label_matrix[i][j] = 1
        else :
          label_matrix[i][j] = 2

    return label_matrix

create_labels(img).shape

"""## **Implementing the perceptron’s error-correction rule**"""

''' Function that returns the prediciton given the weight vector and the input vector for an image '''
def prediction_img(w,x):
    if (np.dot(w,x) > 0) :
        return 1
    else :
        return 2

''' Function to compute the average error for an image '''
def error(image,label_matrix,w):
  s = 0
  for i in range(len(image)):
    for j in range(len(image[0])):
      pixel = np.array([image[i][j],1])
      s+= abs(label_matrix[i][j]-prediction_img(w,pixel))
  return s


''' Implementation of the perceptron's error-correction rule'''
def error_correction(img, label_matrix, eta, w):

  image = img[0]
  line  = len(image)
  row   = len(image[0])

  epoch = 0
  
  w_tmp = np.add(w,np.array([1,1]))


  while(np.allclose(w,w_tmp)==False):
    w_tmp = w
    print("Epoch number",epoch)
    print("w=",w)
    print("error=",error(image,label_matrix,w))
    for i in range(line):
      for j in range(row):
        pixel = np.array([image[i][j],1])
        if (label_matrix[i][j]-prediction_img(w,pixel) == 1):
          w = w + eta*pixel
        elif (label_matrix[i][j]-prediction_img(w,pixel) == -1):
          w = w - eta*pixel
            
    epoch+=1
  
  return w, epoch


''' Test '''
img=tiilab.imz2mat("landsattarasconC4.ima") 
image = img[0]
label_matrix=create_labels(img)
eta=0.01
w=np.array([0,0])

error_correction(img, label_matrix, eta, w)