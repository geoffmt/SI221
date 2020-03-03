from random import *
import numpy as np 
import matplotlib.pyplot as plt



def generate_data_set(sigma):
	u = np.zeros(200)
	v = random.sample(range(1, 200), 50)
	z = []
	mean = []
	cov = [[sigma,0],[0,sigma]]
	x =[]
	for j in range(len(v)):
		a = v[j]
		z.append(u[a])

	for i in range(200):
		if (!z[i]):
			mean = [-1,0]
		elif (z[i]):
			mean = [1,0]
		x.append(np.random.multivariate_normal(mean, cov))

	# generate the couples 
	'''
	result = []
	for i in range(200):
		result.append([x[i], y[i]])
	'''
	result=[]
	result.append([x],[y])
	return result



def perceptron():
	sigma = 0.05
	data_set = generate_data_set(sigma)

	# generate w
	w = np.zeros(200)
	converged = 0
	while (!converged && other_thing_to_be_determined):
		if (data_set[0][1]==1 && np.dot(w, data_set[0])>0):
			converged = 1
		if (data_set[0][1]==0 && np.dot(w, data_set[0])<=0):
			converged = 1
		if (data_set[0][1]==0 && np.dot(x, data_set[0])>0):
			converged = 0
			adjustment_parameter = 0.1
			w = w-adjustment_parameter*data_set[0]





