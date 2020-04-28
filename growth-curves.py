"""	Growth Curves 
	Copyright (C) 2020 Evren Turan

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>
"""
import numpy as np
from scipy.optimize import least_squares

# Variables

file_name  = "Example_1.txt"
model_list = ['Gompertz', 'Quadratic','Logistic','vonBertalanffy','monomolecular']
#
# Open file, load title and read in data
file = open(file_name,"r")
Title = file.readline()
raw_data = file.readlines()

# Double-nested list to get make list of strings to matrix
# the first column must be x data, second column y data
data =([[float(val) for val in line.split()] for line in raw_data[1:]])

# todo: catch if the list contains a blank line... probably a trailing line.
data = np.array(data) 
x_exp=data[:,0]
y_exp=data[:,1]

shape = data.shape

# Model

# Quadratic			3 variables: a,b,c

def Quadratic(x):
	x=abs(x)
	y_pred = x[0]*x_exp**2 + x[1]*x_exp +x[2]
	
	return np.array(y_pred)

# Gompertz 3a 	    3 Variables: a,b,c	
def Gompertz(x):
	x=abs(x)
	y_pred = np.exp(-x[2]*x_exp)*x[1]*-1
	y_pred = x[0]*np.exp(y_pred)
	return y_pred

# Logistic 3a 	    3 Variables: a,b,c	
def Logistic(x):
	x[0:1]=abs(x[0:1])
	y_pred = np.exp(-x[1]*(-x[2]+x_exp))
	y_pred = x[0]/(1+y_pred)
	return y_pred

# von Bertalanffy 3

def vonBertalanffy(x):
	x[0:1]=abs(x[0:1])
	y_pred = 1-np.exp(-x[1]*(-x[2]+x_exp))
	y_pred = x[0]*(y_pred**3)
	return y_pred

# monomolecular
def monomolecular(x):
	x=abs(x)
	y_pred = np.exp(-x[1]*x_exp)
	y_pred = x[0]*(1-x[3]*y_pred)
	return y_pred


def model_y(x):
	if model=="Quadratic":
		y_pred=Quadratic(x)
	elif model=="Gompertz":
		y_pred=Gompertz(x)
	elif model=="Logistic":
		y_pred=Logistic(x)
	elif model=="vonBertalanffy":
		y_pred=vonBertalanffy(x)
	elif model=="monomolecular":
		y_pred=monomolecular(x)
	return y_pred
def calc_res(x):
	# selects model and returns residual
	y_pred=model_y(x)
	res = (y_exp-y_pred)
	return np.array(res)

x0 = np.ones(3)
i=0
perf=np.zeros((len(model_list)))
for model in model_list:

	# call the least square solver
	result=least_squares(calc_res,x0,method='trf')
	perf[i]=result.cost
	i += 1
	if result.success :
		print ('Model: %s 	Residual: %d' % (model,result.cost))
		y_pred=model_y(result.x)
		mat = np.matrix([x_exp,y_pred])
		mat = np.rot90(mat)
		with open('./Output/'+model+'_results.txt','wb') as f:
			for line in mat:
				np.savetxt(f, line, fmt='%.2f')
	else:
		print ('Model: %s 	Residual: Convergence Failed' % (model))




