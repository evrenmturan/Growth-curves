# Growth Curves
# By Evren Turan

import numpy as np
from scipy.optimize import least_squares

# Variables

file_name  = "Example_1.txt"
model = "Quadratic"
#
# Open file, load title and read in data
file = open(file_name,"r")
Title = file.readline()
raw_data = file.readlines()

# Double-nested list to get make list of strings to matrix
# the first column must be x data, second column y data
data =np.array([[float(val) for val in line.split()] for line in raw_data[1:]])
x_exp=data[:,0]
y_exp=data[:,1]

shape = data.shape

# Model

# Quadratic		3 variablesL a,b,c
# y = a*x^2 + bx +c

def Quadratic(x):
	x=abs(x)
	y_pred = x[0]*x_exp**2 + x[1]*x_exp +x[2]
	
	return np.array(y_pred)

def calc_res(x):
	if model=="Quadratic":
		y_pred=Quadratic(x)
		res = (y_exp-y_pred)

	return np.array(res)
x0 = np.ones(3)*0

# call the least square solver, the Levenberg-Marquardt algorithm as implemented in MINPACK is used
result=least_squares(calc_res,x0,method='lm')

if result.success ==False:
	# try the optimisation again using a much more expensive method
	result=least_squares(calc_res,x0,method='trf')

if result.success :
	print 'Sucess'
	y_pred = Quadratic(result.x)
	mat = np.matrix([x_exp,y_pred])
	mat = np.rot90(mat)
	with open(model+'_results.txt','wb') as f:
		for line in mat:
			np.savetxt(f, line, fmt='%.2f')
	



