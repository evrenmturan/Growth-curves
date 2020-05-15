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
import sys
import matplotlib.pyplot as plt
from math import copysign
from scipy import odr
from scipy import stats
import uncertainties as unc
import uncertainties.unumpy as unp
from scipy.linalg import svd
from scipy.optimize import least_squares
from math import pi
import numpy as np
import matplotlib
matplotlib.use('Agg')
# Variables
file_input = "Input.txt"
file = open(file_input, "r")

file.readline()  # line is comment
file_name = file.readline()

file.readline()  # line is comment
regression_type = file.readline()  # 'OLR' or 'ODR' 

file.readline() # line is comment
loss_fn = file.readline()  # 'linear' or 'soft_l1'. 'soft_l1’ deals with outliers better
file.readline() # line is comment
plot_points = file.readline()  # 'linear' or 'soft_l1'. 'soft_l1’ deals with outliers better
file.close()
# remove whitespace
file_name = file_name.strip()
loss_fn = loss_fn.strip()
regression_type=regression_type.strip()
plot_points=plot_points.strip()
plot_points=int(plot_points)
# Open file, load title and read in data
file = open(file_name, "r")
Title = file.readline()

model_list = ['Gompertz', 'Quadratic', 'Logistic', 'vonBertalanffy', 'monomolecular', 'ChapmanRichards', 'HeLegendre', 'Korf', 'Weibull', 'MichaelisMenten', 'NegativeExponential',
              'Power', 'MorganMercerFlodin', 'UnifiedRichards4', 'logistic_2', 'linear', 'vonBertalanffy2','Richards4'] 

plot_range = [float(val) for val in file.readline().split()]
raw_data = file.readlines()
file.close()
# Double-nested list to get make list of strings to matrix
# the first column must be x data, second column y data
data = ([[float(val) for val in line.split()] for line in raw_data[1:]])

# todo: catch if the list contains a blank line... probably a trailing line.
data = np.array(data)
x_exp = data[:, 0]
y_exp = data[:, 1]
# Calculating the Corrected sample deviation
sx_exp = sum((x_exp-x_exp.mean()) ** 2)
sx_exp2 = (sx_exp/(x_exp.size-1))
sy_exp = sum((y_exp-y_exp.mean()) ** 2)
sy_exp2 = (sy_exp/(y_exp.size-1))
shape = data.shape
certain = True
# Model

def Richards6(param,xp):
	# A K Q B M V
	# 0 1 2 3 4 5
	num = param[1]-param[0]
	if certain:
		den = np.exp(-param[3]*(xp-param[4]))
	else:
		den = unp.exp(-param[3]*(xp-param[4]))
	den = (1+ den*param[2])**(1/param[5])
	y_pred = param[0]+num/den
	return y_pred

def vonBertalanffy2(param, xp):
    if certain:
        y_pred = param[0]*((1-np.exp(-param[1]*xp))**3)
    else:
        y_pred = param[0]*((1-unp.exp(-param[1]*xp))**3)
    return y_pred


def linear(param, xp):
    y_pred = param[0]+param[1]*xp
    return y_pred


def logistic_2(param, xp):
    if certain:
        y_pred = -1 + ((1+param[0])*np.exp(param[1]*xp)
                       )/(param[0]+np.exp(param[1]*xp))
    else:
        y_pred = -1 + ((1+param[0])*unp.exp(param[1]*xp)
                       )/(param[0]+unp.exp(param[1]*xp))
    return y_pred


def UnifiedRichards4(param, xp):

    A = param[0]
    d = param[1]
    W = param[2]
    K = param[3]
    # d cannot = 1
    if (abs(d-1) < 1e-12):
        dden = 0.367861
    else:
        dden = d ** (d/(1.0-d))  

    if certain:
        part1 = np.exp(-K*xp/dden)
    else:
        part1 = unp.exp(-K*xp/dden)
    part2 = (W/A)**(1-d) - 1
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
		replace = copysign(1e-12,param[1]-1)
		param[1] = param[1]+1
	else:
		y_pred = (1 + (param[1]-1)*y_pred) ** (1/(1-param[1]))
	y_pred = y_pred*param[0]
	return y_pred
# 3 variables: a,b,c


def MorganMercerFlodin(param, xp):
    y_pred = (param[0]*(xp**param[2]))/(param[1]+xp**param[2])
    return y_pred


def Power(param, xp):
    y_pred = param[2]+param[0]*(xp**param[1])
    return y_pred


def NegativeExponential(param, xp):
    if certain:
        y_pred = param[0]*(1-np.exp(-param[1]*(-param[2]+xp)))
    else:
        y_pred = param[0]*(1-unp.exp(-param[1]*(-param[2]+xp)))

    return y_pred


def MichaelisMenten(param, xp):
    y_pred = param[2]+(param[0]-param[2])*xp/(param[1]+xp)
    return y_pred


def Weibull(param, xp):
    if certain:
        y_pred = -param[1]*(xp**param[2])
        y_pred = param[0]*(1-np.exp(y_pred))
    else:
        y_pred = -param[1]*(xp**param[2])
        y_pred = param[0]*(1-unp.exp(y_pred))
    return y_pred


def Korf(param, xp):
    if certain:
        y_pred = param[0]*np.exp(-param[1]*(xp**(-1*param[2])))
    else:
        y_pred = param[0]*unp.exp(-param[1]*(xp**(-1*param[2])))
    return y_pred


def HeLegendre(param, xp):
    num = param[0]*param[1]
    den = param[1]+(xp**(-param[2]))
    y_pred = num/den
    return y_pred


def ExtremeValue(param, xp):
    if certain:
        y_pred = -np.log(param[2]+param[1]*xp)
        y_pred = param[0]*(1+np.exp(y_pred))
    else:
        y_pred = -unp.log(param[2]+param[1]*xp)
        y_pred = param[0]*(1+unp.exp(y_pred))
    return y_pred


def Quadratic(x, xp):
    y_pred = x[0]*xp**2 + x[1]*xp + x[2]
    y_pred = np.array(y_pred)
    # parameter unceratinty is okay.
    return y_pred

# Gompertz 3a 	    3 Variables: a,b,c


def Gompertz(x, xp):
    if certain:
        y_pred = np.exp(-x[2]*xp)*x[1]*-1
        y_pred = x[0]*np.exp(y_pred)
    else:
        y_pred = unp.exp(-x[2]*xp)*x[1]*-1
        y_pred = x[0]*unp.exp(y_pred)
    return y_pred

# Logistic 3a 	    3 Variables: a,b,c


def Logistic(x, xp):
    if certain:
        y_pred = np.exp(-x[1]*(-x[2]+xp))
        y_pred = x[0]/(1+y_pred)
    else:
        y_pred = unp.exp(-x[1]*(-x[2]+xp))
        y_pred = x[0]/(1+y_pred)
    return y_pred

# von Bertalanffy 3


def vonBertalanffy(x, xp):
    if certain:
        y_pred = 1-np.exp(-x[1]*(-x[2]+xp))
        y_pred = x[0]*(y_pred**3)
    else:
        y_pred = 1-unp.exp(-x[1]*(-x[2]+xp))
        y_pred = x[0]*(y_pred**3)
    return y_pred

# monomolecular


def monomolecular(x, xp):

    if certain:
        y_pred = np.exp(-x[1]*xp)
        y_pred = x[0]*(1-x[2]*y_pred)
    else:
        y_pred = unp.exp(-x[1]*xp)
        y_pred = x[0]*(1-x[2]*y_pred)
    return y_pred

# Chapman-Richards


def ChapmanRichards(x, xp):

    if certain:
        y_pred = 1-np.exp(-x[1]*xp)
        y_pred = x[0]*(y_pred**x[2])
    else:
        y_pred = 1-unp.exp(-x[1]*xp)
        y_pred = x[0]*(y_pred**x[2])
    return y_pred


def model_y(param, xp):

    y_pred = eval(model)(param, xp)
    return y_pred


def calc_res(param):
    # selects model and returns residual
    y_pred = model_y(param, x_exp)
    res = (y_exp-y_pred)
    return res
def numder(param):
    dfdy=np.zeros([len(param),len(x_exp)])
    for i in range(len(param)):
        param_new = param
        param_new[i] = param[i]*(1+1e-6)
        res1 = (calc_res(param_new))
        param_new[i] = param[i]*(1-1e-6)
        res2 = (calc_res(param_new))
        dfdy[i,:] = (res1-res2)/param[i]*(1+2e-6)
    return dfdy
def conf_curve(x, x_exp, y_exp, param, model, y_pred, y_points, alpha=0.05):
    # X = points for plotting
    # param = fitted params
    # model is model name
    # assuming standard alpha, i.e. p=0.95
    Ns = x_exp.size  # sample size
    Nvar = len(param)

    # from t distribution
    quant = stats.t.ppf(1.0-alpha/2.0, Ns - Nvar)
    res = (y_exp-y_pred)

    # std deviation of point
    stdn = np.sqrt(1.0/(Ns - Nvar) * np.sum(res**2))

    sx = (x - x_exp.mean()) ** 2
    sxdev = np.sum((x_exp - x_exp.mean()) ** 2)

    # therefore enough has been calculated to give
    # the confidence intervals

    dy = quant*stdn*np.sqrt(1.0 + (1.0/Ns) + (sx/sxdev))
    low, upp = y_points - dy, y_points + dy
    return low, upp


x0 = np.ones(3)
i = 0
perf = np.zeros((len(model_list)))
AIC = np.ones((len(model_list)))*100
AIC_vec = np.ones((len(model_list)))*100
R2vec = np.zeros((len(model_list)))
R2barvec = np.zeros((len(model_list)))
SST = np.sum((y_exp-y_exp.mean())**2)  # defining total sum of squares


x_points = np.linspace(plot_range[0], plot_range[1], plot_points)

for model in model_list:
    
    model_name = model
    x0 = np.ones(3)
    four_param = False
    # call the least square solver
    # defining the bounds
    uplow = ([0, 0, -np.inf], np.inf)
    if (model_name == 'ChapmanRichards' or model_name == 'monomolecular' or model_name == 'Gompertz' or model_name == 'HeLegendre' or model_name == 'Korf' or model_name == 'Weibull'):
        uplow = ([0, 0, 0], np.inf)
    if (model_name == 'Power'):
        uplow = ([0, -np.inf, -np.inf], np.inf)

    if (model_name == 'UnifiedRichards4'):
        x0 = np.ones(4)
        # A d k W	1 2 3 4
        x0[0] = max(x_exp)*1.1
        x0[1] = 2
        uplow = ([0.0, 0, 0.0, 0.0], [np.inf, np.inf, np.inf, max(y_exp)])

    if (model_name == 'Richards4'):
        x0 = np.ones(4)
        # A d k T 	1 2 3 4
        # d>-1
        x0[0] = max(x_exp)*1.1
        uplow = ([0.0, -1.0, 0.0, 0.0], [np.inf, 2, np.inf, np.inf])


        
    if(model_name == 'logistic_2' or model_name == 'vonBertalanffy2'):
        x0 = np.ones(2)
        uplow = ([0, 0], np.inf)
    if (model_name == 'linear'):
        x0 = np.ones(2)
        uplow = (-np.inf, np.inf)
    if (model_name == 'Richards6'):
        x0 = np.ones(6)
        x0[0] = 0
        x0[1] = max(y_exp)
        uplow = ([-np.inf, -np.inf, -np.inf, -np.inf, -np.inf, 0.0, ], np.inf)

    
    result = least_squares(calc_res, x0, method='trf', jac='3-point',
                           max_nfev=1000*len(x0), bounds=uplow, loss=loss_fn)
    if not result.success:
        # try again with more accurate jacobian and longer max function evals.
        result = least_squares(calc_res, x0, method='trf', jac='3-point',
        max_nfev=1000*len(x0)*len(x0), bounds=uplow, loss=loss_fn)

    success = result.success
    fit_param = result.x
    if regression_type == 'ODR':
        path='./Output/ODR/'+model
        model_instance = odr.Model(model_y)
        # requires model x values, yvalues, and standard deviations
        data_instance = odr.Data(x_exp, y_exp, wd=1./sx_exp2, we=1./sy_exp2)
        # using the OLS results as the initial value
        myodr = odr.ODR(data_instance, model_instance, beta0=result.x)
        odr.ODR.set_job(myodr,fit_type=0, deriv=1) # ODR = type 0, central dif. = deriv 1
        myoutput = myodr.run()       
        if myoutput.info<4:
            success = True
            fit_param = myoutput.beta
        else:
            success = False
    else:
        path='./Output/OLS/'+model
    if success:

        if regression_type=='OLS':
            
            # estimating pcov using method from minpack.py
            # Do Moore-Penrose inverse discarding zero singular values.
            _, s, VT = svd(result.jac, full_matrices=False)
            threshold = np.finfo(float).eps * max(result.jac.shape) * s[0] 
            s = s[s > threshold]
            VT = VT[:s.size]
            pcov = np.dot(VT.T / s**2, VT)
            cost = result.cost * 2
            s_sq = cost / (y_exp.size - x0.size)
            pcov = pcov * s_sq
            sd_beta = unc.correlated_values(fit_param, pcov)
            #U, s, Vh = svd(result.jac, full_matrices=False)
            #threshold = 1e-14  # anything below machine error*100 = 0
            #s = s[s > threshold]
            #Vh = Vh[:s.size]
            #pcov = np.dot(Vh.T / s**2, Vh)
        else:
            sd_beta=unp.uarray(fit_param,myoutput.sd_beta)
        perf = (sum(calc_res(fit_param)**2))  # (sum (res^2)*0.5)*2
        s2 = perf/x_exp.size
        # leaving out np.log(2*pi)
        logL = -(x_exp.size)/2*(np.log(2*pi)+np.log(s2)+1)
        AIC = 2*len(fit_param)-2*logL
        AIC = AIC + (2*len(fit_param)*(len(fit_param)+1)) / \
            (x_exp.size-len(fit_param)-1)
        # for listing later
        AIC_vec[i] = AIC
        SSR = sum(calc_res(fit_param)**2)  # calculating the SSR

        R2vec[i] = 1 - (SSR)/SST
        R2barvec[i] = 1 - (1-R2vec[i])*(x_exp.size-1) / \
            (x_exp.size-1-len(fit_param))
        i = i+1
        print('Model: %s AIC: %d R2 %4.2f, R2bar %4.2f' %
              (model, AIC, R2vec[i-1], R2barvec[i-1]))
        certain = True
        y_pred = model_y(fit_param, x_exp)
        certain = False
        if len(fit_param) == 6:
             a, b, c, d, e, ff=sd_beta
             y_unc = model_y(np.array(sd_beta), x_points)

        if len(fit_param) == 2:
            a,b =sd_beta
            y_unc = model_y(np.array(sd_beta), x_points)
            c = ''
            d = ''
        if len(fit_param) == 3:
            a,b,c = sd_beta
            y_unc = model_y(np.array(sd_beta), x_points)
            d = ''

        if len(fit_param) == 4:
            
            a,b,c,d = sd_beta
            y_unc = model_y(np.array(sd_beta), x_points)
        
        print('Uncertainty')

        print('a: ' + str(a))
        print('b: ' + str(b))
        print('c: ' + str(c))
        print('d: ' + str(d))
        if len(fit_param) == 6:
            print('e: ' + str(e))
            print('f: ' + str(ff))
        # for graphing

        y_points = unp.nominal_values(y_unc)
        std = unp.std_devs(y_unc)

        certain = True
        low, upp = conf_curve(x_points, x_exp, y_exp,
                              fit_param, model, y_pred, y_points, alpha=0.05)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(x_points, y_points, label='Model')
        ax.plot(x_points, low, 'k--', label='95% prediction band')
        ax.plot(x_points, upp, 'k--')
        ax.plot(x_exp, y_exp, 'bo', label='Experimental points')

        # uncertainty lines (95% confidence)
        ax.plot(x_points, y_points - 1.96*std,
                c='orange', label='95% confidence band')

        ax.plot(x_points, y_points + 1.96*std, c='orange')
        ax.set_xlim([0, None])
        ax.set_ylim([0, None])
        ax.legend(loc='best')

        fig.savefig(path+'.png')
        # printing out
        file_out = open(path+'_results.txt', 'w')
        f = file_out
        file_out.write("Parameters: A, B & C	\n\n")
        file_out.write('a: ' + str(a))
        file_out.write('\nb: ' + str(b))
        file_out.write('\nc: ' + str(c))
        file_out.write('\nd: ' + str(d))
        if len(fit_param) == 6:
            file_out.write('\ne: ' + str(e))
            file_out.write('\nf: ' + str(ff))
        file_out.write("\n\nPlotting Data: Predicted \n\n")

        mat = np.matrix([x_points, y_points])
        mat = np.rot90(mat)
        for line in mat:
            np.savetxt(f, line, fmt='%.2f')

        file_out.write("\n\nPlotting Data: Confidence Interval Upper	\n\n")
        mat = np.matrix([x_points, upp])
        mat = np.rot90(mat)
        for line in mat:
            np.savetxt(f, line, fmt='%.2f')

        file_out.write("\n\nPlotting Data: Confidence Interval Lower	\n\n")
        mat = np.matrix([x_points, low])
        mat = np.rot90(mat)
        for line in mat:
            np.savetxt(f, line, fmt='%.2f')
        file_out.close()
    else:
        print('Model: %s 	Residual: Convergence Failed' % (model))
        AIC_vec[i] = np.inf
        R2vec[i] = float('nan')
        R2barvec[i] = float('nan')
        i=i+1
        fig = plt.figure()
        fig.savefig(path+'.png') # to produce blank figure
        file_out = open(path+'_results.txt', 'w')
        file_out.write('Model regression failed.')
        file_out.close()

# Final organising for outputs
# delta AIC is more useful than AIC
AIC_vec = AIC_vec-min(AIC_vec)
if regression_type=='OLS':
    path='./Output/OLS/Summary.txt'
elif regression_type=='ODR':
    path='./Output/ODR/Summary.txt'

file_out = open(path, 'w')
file_out.write('Summary of outputs\n')
file_out.write('Input:	%s\n' % file_name)
file_out.write('Loss Method:	%s\n\n' % loss_fn)

file_out.write('Model results in order of increasing deltaAIC:\n')
sorted_list = [model_list for _,
               model_list in sorted(zip(AIC_vec, model_list))]
sorted_AIC = [AIC_vec for _, AIC_vec in sorted(zip(AIC_vec, AIC_vec))]
sorted_R2 = [R2vec for _, R2vec in sorted(zip(AIC_vec, R2vec))]
sorted_R2bar = [R2barvec for _, R2barvec in sorted(zip(AIC_vec, R2barvec))]

file_out.write("%-30s	%4s	%4s	%4s\n" % ('Model', 'AIC', 'R2', 'R2bar'))

for i in range(len(sorted_list)):
    file_out.write("%-30s	%4.4f	%4.4f	%4.4f\n" %
                   (sorted_list[i], sorted_AIC[i], sorted_R2[i], sorted_R2bar[i]))
