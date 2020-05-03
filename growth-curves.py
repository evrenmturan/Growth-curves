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
from math import copysign
# Variables

file_name  = "Example_1.txt"
loss_fn = 'linear' # 'linear' or 'soft_l1'. 'soft_l1’ deals with outliers better
# Open file, load title and read in data
file = open(file_name,"r")
Title = file.readline()

model_list = ['Gompertz', 'Quadratic','Logistic','vonBertalanffy','monomolecular','ChapmanRichards','HeLegendre','Korf','Weibull','MichaelisMenten','NegativeExponential',\
	'Power','MorganMercerFlodin','Richards4','UnifiedRichards4']

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
# 4 variable
def UnifiedRichards4(param,xp):

	A=param[0]
	d=param[1]
	W=param[2]
	K=param[3]
	# d cannot = 1
	if (abs(d-1) < 1e-12):
		replace = copysign(1e-8,d-1)
		d = replace+1
		dden=1
	else:
		dden = d ** (d/(1.0-d)) # ~= 1 **(inf)

	if certain:
		part1 = np.exp(-K*xp/dden)
	else:
		part1 = unp.exp(-K*xp/dden)
	part2 = (W/A)**(1-d) -1
	y_pred = (1+part1*part2) ** (1/(1-d))
	y_pred = A*(y_pred)
	return y_pred
def Richards4(param,xp):
	# A d k T 	1 2 3 4
	# d>-1
	
	# param[1] cannot equal 1 due to division by zero. 
	if certain:
		y_pred = np.exp(-param[2]*(xp-param[3]))
	else:
		y_pred = unp.exp(-param[2]*(xp-param[3]))
	if (abs(param[1]-1) < 1e-12):
		y_pred=1 #(1+ ~0)**(inf)
	else:
		y_pred = (1 + (param[1]-1)*y_pred) ** (1/(1-param[1]))
	y_pred = y_pred*param[0]
	return y_pred
# 3 variables: a,b,c
def MorganMercerFlodin(param,xp):
	y_pred = (param[0]*(xp**param[2]))/(param[1]+xp**param[2])
	return y_pred
def Power(param,xp):
	y_pred = param[2]+param[0]*(xp**param[1])
	return y_pred
def NegativeExponential(param,xp):
	if certain:
			y_pred = param[0]*(1-np.exp(-param[1]*(-param[2]+xp)))
	else:
			y_pred = param[0]*(1-unp.exp(-param[1]*(-param[2]+xp)))

	return y_pred

def MichaelisMenten(param,xp):
	y_pred = param[2]+(param[0]-param[2])*xp/(param[1]+xp)
	return y_pred
def Weibull(param,xp):
	if certain:
		y_pred = -param[1]*(xp**param[2])
		y_pred = param[0]*(1-np.exp(y_pred))
	else:
		y_pred = -param[1]*(xp**param[2])
		y_pred = param[0]*(1-unp.exp(y_pred))
	return y_pred
def Korf(param,xp):
	if certain:
			y_pred = param[0]*np.exp(-param[1]*(xp**(-1*param[2])))
	else:
			y_pred = param[0]*unp.exp(-param[1]*(xp**(-1*param[2])))
	return y_pred
def HeLegendre(param,xp):
	num = param[0]*param[1]
	den = param[1]+(xp**(-param[2]))
	y_pred = num/den
	return y_pred

def ExtremeValue(param,xp):
	if certain:
		y_pred = -np.log(param[2]+param[1]*xp)
		y_pred = param[0]*(1+np.exp(y_pred))
	else:
		y_pred = -unp.log(param[2]+param[1]*xp)
		y_pred = param[0]*(1+unp.exp(y_pred))
	return y_pred


def Quadratic(x,xp):
	y_pred = x[0]*xp**2 + x[1]*xp +x[2]
	y_pred=np.array(y_pred)
	#parameter unceratinty is okay.
	return y_pred

# Gompertz 3a 	    3 Variables: a,b,c	
def Gompertz(x,xp):
	if certain:
		y_pred = np.exp(-x[2]*xp)*x[1]*-1
		y_pred = x[0]*np.exp(y_pred)
	else:
		y_pred = unp.exp(-x[2]*xp)*x[1]*-1
		y_pred = x[0]*unp.exp(y_pred)
	return y_pred

# Logistic 3a 	    3 Variables: a,b,c	
def Logistic(x,xp):
	if certain:
		y_pred = np.exp(-x[1]*(-x[2]+xp))
		y_pred = x[0]/(1+y_pred)
	else:
		y_pred = unp.exp(-x[1]*(-x[2]+xp))
		y_pred = x[0]/(1+y_pred)
	return y_pred

# von Bertalanffy 3

def vonBertalanffy(x,xp):
	if certain:
		y_pred = 1-np.exp(-x[1]*(-x[2]+xp))
		y_pred = x[0]*(y_pred**3)
	else:
		y_pred = 1-unp.exp(-x[1]*(-x[2]+xp))
		y_pred = x[0]*(y_pred**3)
	return y_pred

# monomolecular
def monomolecular(x,xp):

	if certain:
		y_pred = np.exp(-x[1]*xp)
		y_pred = x[0]*(1-x[2]*y_pred)
	else:
		y_pred = unp.exp(-x[1]*xp)
		y_pred = x[0]*(1-x[2]*y_pred)
	return y_pred

# Chapman-Richards
def ChapmanRichards(x,xp):

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
AIC_vec = np.ones((len(model_list)))*100
R2vec = np.zeros((len(model_list)))
SST = np.sum((y_exp-y_exp.mean())**2) # defining total sum of squares

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
	x0 = np.ones(3)
	four_param = False
	# call the least square solver
	# defining the bounds 
	uplow = ([0, 0, -np.inf], np.inf)
	if (model_name=='ChapmanRichards' or model_name=='monomolecular' or model_name=='Gompertz' or model_name=='HeLegendre' or model_name=='Korf' or model_name=='Weibull'):
		uplow = ([0, 0, 0], np.inf)
	if (model_name == 'Power'):
		uplow = ([0, -np.inf, -np.inf], np.inf)
	if (model_name == 'UnifiedRichards4') :
		x0 = np.ones(4)*1
		# A d k W	1 2 3 4
		four_param=True
		x0[0]=max(x_exp)*1.1
		x0[1]=2
		uplow = ([0.0, 0, 0.0,0.0], [np.inf, np.inf, np.inf,max(y_exp)])
	if (model_name == 'Richards4') :
		x0 = np.ones(4)*1
		# A d k T 	1 2 3 4
		# d>-1
		x0[0]=max(x_exp)*1.1
		four_param=True
		uplow = ([0.0, -1.0, 0.0,0.0], [np.inf, 2, np.inf,np.inf])

	result=least_squares(calc_res,x0,method='trf',jac='3-point',max_nfev=1000*len(x0),bounds=uplow,loss=loss_fn)
	if not result.success:
	# try again with more accurate jacobian and longer max function evals.
			result=least_squares(calc_res,x0,method='trf',jac='3-point',max_nfev=1000*len(x0)*len(x0),bounds=uplow,loss=loss_fn)

	# Estimating covariance matrix form the jacobian
	# as H = 2*J^TJ and cov = H^{-1}
	U, s, Vh = svd(result.jac, full_matrices=False)
	threshold = 1e-14 #anything below machine error*100 = 0
	s = s[s > threshold]
	Vh = Vh[:s.size]
	pcov = np.dot(Vh.T / s**2, Vh)
	perf=(sum(calc_res(result.x)**2)) # (sum (res^2)*0.5)*2
	s2=perf/x_exp.size
	logL = -(x_exp.size)/2*(np.log(2*pi)+np.log(s2)+1) # leaving out np.log(2*pi)
	AIC = 2*len(result.x)-2*logL
	AIC = AIC+ (2*len(result.x)*(len(result.x)+1))/(x_exp.size-len(result.x)-1)
	# for listing later
	AIC_vec[i]=AIC
	res = sum(calc_res(result.x)**2)

	R2vec[i] = 1- (res)/SST
	i=i+1
	# TODO: After indentifying the best AIC the remainder should be scaled: delta AICc = AIC - AICmin.

	x_points = np.linspace(plot_range[0], plot_range[1], 20)
	if result.success :
		print ('Model: %s 	Residual: %d 	AIC: %d R2vec %4.2f' % (model,res,AIC,R2vec[i-1]))
		certain=True
		y_pred=model_y(result.x,x_exp)
		certain=False
		if len(result.x)==3:
			a,b,c = unc.correlated_values(result.x, pcov)
			y_unc = model_y(np.array([a,b,c]),x_points)
			d=''
			

		if len(result.x)==4:
			a,b,c,d = unc.correlated_values(result.x, pcov)
			y_unc = model_y(np.array([a,b,c,d]),x_points)
		print('Uncertainty')
			
		print('a: ' + str(a))
		print('b: ' + str(b))
		print('c: ' + str(c))
		print('d: ' + str(d))
		# for graphing
		
		
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
		file_out.write('\nd: ' + str(d))
	
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

file_out = open('./Output/Summary.txt', 'w')
file_out.write('The models given in order of increasing AIC:\n')
sorted_list = [model_list for _,model_list in sorted(zip(AIC_vec,model_list))]     
for item in sorted_list:
        file_out.write("%s,  " % item)
                                 

file_out.write('\n%s has the lowest AIC of %4.2e\n' %(sorted_list[0],min(AIC_vec)))

file_out.write('\n\nThe models given in order of decreasing R2: \n')
sorted_list = [model_list for _,model_list in sorted(zip(R2vec,model_list),reverse=True)]                                      
for item in sorted_list:
        file_out.write("%s,  " % item)
file_out.write('\n%s has the highest R2vec of %4.2f\n' %(sorted_list[0],max(R2vec)))