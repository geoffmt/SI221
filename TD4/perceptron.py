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


def perceptron():
	