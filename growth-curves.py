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

# IMPORTANT NOTE!
# Errors will occur in this code if one of the fitted parameters wraps around 0.
# i.e. if a = 0.5 +/- 0.6 OR b = -1 +/- 1.5 
# then the quoted confidence is incorrect 

import matplotlib
matplotlib.use('Agg')
import numpy as np
from math import pi
from scipy.optimize import least_squares
from scipy.linalg import svd
import uncertainties.unumpy as unp
import uncertainties as unc
import matplotlib.pyplot as plt  
from scipy import stats
# Variables

file_name  = "Example_1.txt"
model_list = ['Gompertz', 'Quadratic','Logistic','vonBertalanffy','monomolecular','ChapmanRichards']
#
# Open file, load title and read in data
file = open(file_name,"r")
Title = file.readline()
plot_range = [float(val) for val in file.readline().split()] 
raw_data = file.readlines()
file.close()
# Double-nested list to get make list of strings to matrix
# the first column must be x data, second column y data
data =([[float(val) for val in line.split()] for line in raw_data[1:]])

# todo: catch if the list contains a blank line... probably a trailing line.
data = np.array(data) 
x_exp=data[:,0]
y_exp=data[:,1]

shape = data.shape
certain=True
# Model

# Quadratic			3 variables: a,b,c
def Quadratic(x,xp):
	y_pred = x[0]*xp**2 + x[1]*xp +x[2]
	y_pred=np.array(y_pred)
	#parameter unceratinty is okay.
	return y_pred

# Gompertz 3a 	    3 Variables: a,b,c	
def Gompertz(x,xp):
	x=abs(x)
	if certain:
		y_pred = np.exp(-x[2]*xp)*x[1]*-1
		y_pred = x[0]*np.exp(y_pred)
	else:
		y_pred = unp.exp(-x[2]*xp)*x[1]*-1
		y_pred = x[0]*unp.exp(y_pred)
	return y_pred

# Logistic 3a 	    3 Variables: a,b,c	
def Logistic(x,xp):
	x[0:1]=abs(x[0:1])
	if certain:
		y_pred = np.exp(-x[1]*(-x[2]+xp))
		y_pred = x[0]/(1+y_pred)
	else:
		y_pred = unp.exp(-x[1]*(-x[2]+xp))
		y_pred = x[0]/(1+y_pred)
	return y_pred

# von Bertalanffy 3

def vonBertalanffy(x,xp):
	x[0:1]=abs(x[0:1])
	if certain:
		y_pred = 1-np.exp(-x[1]*(-x[2]+xp))
		y_pred = x[0]*(y_pred**3)
	else:
		y_pred = 1-unp.exp(-x[1]*(-x[2]+xp))
		y_pred = x[0]*(y_pred**3)
	return y_pred

# monomolecular
def monomolecular(x,xp):
	x=abs(x)
	if certain:
		y_pred = np.exp(-x[1]*xp)
		y_pred = x[0]*(1-x[2]*y_pred)
	else:
		y_pred = unp.exp(-x[1]*xp)
		y_pred = x[0]*(1-x[2]*y_pred)
	return y_pred

# Chapman-Richards
def ChapmanRichards(x,xp):
	x=abs(x)
	if certain:
		y_pred = 1-np.exp(-x[1]*xp)
		y_pred = x[0]*(y_pred**x[2])
	else:
		y_pred = 1-unp.exp(-x[1]*xp)
		y_pred = x[0]*(y_pred**x[2])
	return y_pred


def model_y(x,xp):

	y_pred=eval(model)(x,xp)
	return y_pred
def calc_res(x):
	# selects model and returns residual
	y_pred=model_y(x,x_exp)
	res = (y_exp-y_pred)
	return res

x0 = np.ones(3)
i=0
perf=np.zeros((len(model_list)))
AIC=np.ones((len(model_list)))*100
def conf_curve (x,x_exp,y_exp,param,model,y_pred,y_points,alpha=0.05):
	# X = points for plotting
	# param = fitted params
	# model is model name
	# assuming standard alpha, i.e. p=0.95
	Ns = x_exp.size # sample size
	Nvar =len(param)

	# from t distribution
	quant = stats.t.ppf(1.0-alpha/2.0, Ns - Nvar)
	res = (y_exp-y_pred)

	#std deviation of point
	stdn = np.sqrt(1.0/(Ns - Nvar) * np.sum(res**2))

	sx =(x - x_exp.mean())** 2
	sxdev = np.sum((x_exp - x_exp.mean())** 2)

	# therefore enough has been calculated to give
	# the confidence intervals

	dy = quant*stdn*np.sqrt(1.0+ (1.0/Ns) + (sx/sxdev))
	low, upp = y_points - dy, y_points + dy
	return low,upp

for model in model_list:
	model_name=model
	# call the least square solver
	result=least_squares(calc_res,x0,method='trf',jac='2-point',max_nfev=100*len(x0))
	if not result.success:
	# try again with more accurate jacobian and longer max function evals.
			result=least_squares(calc_res,x0,method='trf',jac='3-point',max_nfev=1000*len(x0)*len(x0))

	result.x[0:1]=abs(result.x[0:1])
	if (model=="monomolecular" or model=='ChapmanRichards'):
		result.x[2]=abs(result.x[2])
	# Estimating covariance matrix form the jacobian
	# as H = 2*J^TJ and cov = H^{-1}
	U, s, Vh = svd(result.jac, full_matrices=False)
	threshold = 1e-14 #anything below machine error*100 = 0
	s = s[s > threshold]
	Vh = Vh[:s.size]
	pcov = np.dot(Vh.T / s**2, Vh)
	perf=(result.cost*2) # (sum (res^2)*0.5)*2
	s2=perf/x_exp.size
	logL = -(x_exp.size)/2*(np.log(2*pi)+np.log(s2)+1) # leaving out np.log(2*pi)
	AIC = 2*len(result.x)-2*logL
	AIC = AIC+ (2*len(result.x)*(len(result.x)+1))/(x_exp.size-len(result.x)-1)

	# TODO: After indentifying the best AIC the remainder should be scaled: delta AICc = AIC - AICmin.

	x_points = np.linspace(plot_range[0], plot_range[1], 20)
	if result.success :
		print ('Model: %s 	Residual: %d 	AIC: %d ' % (model,result.cost,AIC))
		certain=True
		y_pred=model_y(result.x,x_exp)

		a,b,c = unc.correlated_values(result.x, pcov)
		print('Uncertainty')
		print('a: ' + str(a))
		print('b: ' + str(b))
		print('c: ' + str(c))

		# for graphing
		certain=False
		y_unc = model_y(np.array([a,b,c]),x_points)
		y_points=unp.nominal_values(y_unc)
		std = unp.std_devs(y_unc)
		certain=True
		low, upp = conf_curve(x_points,x_exp,y_exp,result.x,model,y_pred,y_points,alpha=0.05)
		
		fig = plt.figure()
		ax = fig.add_subplot(111)
		ax.plot(x_points,y_points,label='Model')
		ax.plot(x_points,low,'k--',label='95% Prediction band')
		ax.plot(x_points,upp,'k--')
		ax.plot(x_exp,y_exp,'bo',label='Experimental points')
		# uncertainty lines (95% confidence)
		ax.plot(x_points, y_points - 1.96*std, c='orange',label='95% Confidence Region')
         
		ax.plot(x_points, y_points + 1.96*std, c='orange')

		ax.legend(loc='best')

		fig.savefig('./Output/'+model+'.png')
		# printing out
		file_out = open('./Output/'+model+'_results.txt','w')
		f=file_out
		file_out.write("Parameters: A, B & C	\n\n")
		file_out.write('a: ' + str(a))
		file_out.write('\nb: ' + str(b))
		file_out.write('\nc: ' + str(c))
	
	
		file_out.write("\n\nPlotting Data: Predicted \n\n")
		#file_out.close()
		mat = np.matrix([x_points,y_points])
		mat = np.rot90(mat)
		#with open('./Output/'+model+'_results.txt','a') as f:
		for line in mat:
			np.savetxt(f, line, fmt='%.2f')
		
		file_out.write("\n\nPlotting Data: Confidence Interval Upper	\n\n")
		mat = np.matrix([x_points,upp])
		mat = np.rot90(mat)
		for line in mat:
			np.savetxt(f, line, fmt='%.2f')

		file_out.write("\n\nPlotting Data: Confidence Interval Lower	\n\n")
		mat = np.matrix([x_points,low])
		mat = np.rot90(mat)
		for line in mat:
			np.savetxt(f, line, fmt='%.2f')
		file_out.close()
	else:
		print ('Model: %s 	Residual: Convergence Failed' % (model))




