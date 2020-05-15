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

model_list_i = ['Gompertz_3a_i', 'monomolecular_i', 'vonBertalanffy_i', 'HeLegendre_i', 'Korf_i', 'logistic_i', 'MMF_i', 'Weibull_i', 'MichaelisMenten_i', 'NegativeExponential_i',
              'Power_i', 'Power2_i', 'ChapmanRichards_i']
#
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

# Common growth models, inverted:


def ChapmanRichards_i(param, y):
    if certain:
        x = np.log(1-((y/param[0])**(1/param[2])))
    else:
        x = unp.log(1-((y/param[0])**(1/param[2])))
    x = -x/param[1]
    return x

def Power_i(param, y):
    x = ((-param[2]+y)/param[0]) ** (1/param[1])
    return x


def Power2_i(param, y):
    x = (-param[1]+(y ** param[2]))/param[0]
    return x


def Richards_inverse(param, y, m):
    # param[0:1]=abs(param[0:1])
    num = ((y/param[0]) ** (1-m)) - 1
    dem = m-1
    if certain:
        x = (-1/param[1])*np.log(num/dem)+param[2]
    else:
        x = (-1/param[1])*unp.log(num/dem)+param[2]
    return x


def NegativeExponential_i(param, y):
    if certain:
        x = (param[1]*param[2] - np.log(-(-param[0]+y)/param[0]))/param[2]
    else:
        x = (param[1]*param[2] - unp.log(-(-param[0]+y)/param[0]))/param[2]
    return x


def vonBertalanffy_i(param, y):
    m = 2/3.0
    x = Richards_inverse(param, y, m)
    return x


def monomolecular_i(param, y):
    # param[2]=abs(param[2])
    m = 0
    x = Richards_inverse(param, y, m)
    return x


def logistic_i(param, y):
    if certain:
        x = (param[1]*param[2]-np.log(-1+param[0]/y))/param[1]
    else:
        x = (param[1]*param[2]-unp.log(-1+param[0]/y))/param[1]
    return x


def Gompertz_3a_i(param, y):
    if certain:
        x = np.log(y/param[0])
        x = np.log(-x/param[1])
        x = -x/param[2]
    else:
        x = unp.log(y/param[0])
        x = unp.log(-x/param[1])
        x = -x/param[2]
    return x

def HeLegendre_i(param, y):
    x = (-param[1]+param[1]*param[0]/y)**(-1/param[2])
    return x


def Korf_i(param, y):
    if certain:
        x = (-np.log(y/param[0])/param[1]) ** (-1/param[2])
    else:
        x = (-unp.log(y/param[0])/param[1]) ** (-1/param[2])

    return x


def MMF_i(param, y):
    x = (-(-param[0]+y)/(param[1]*y)) ** (-1/param[2])
    return x


def Weibull_i(param, y):
    if certain:
        x = -np.log(-(-param[0]+y)/param[0])/param[1]
        x = x ** (-1/param[2])
    else:
        x = -unp.log(-(-param[0]+y)/param[0])/param[1]
        x = x ** (-1/param[2])
    return x


def MichaelisMenten_i(param, y):
    x = (-param[1]*param[2]+param[1]*y)/(param[0]-y)
    return x


def model_x(param, y_points):
    x_points = eval(model)(param, y_points)
    return x_points


def calc_res(param):
    x_pred = model_x(param, y_exp)
    res = x_exp - x_pred
    return res


p0 = np.ones(3)  # initial parameter points
# giving initial guesses
p0[0] = max(y_exp)*1.4

i=0

perf = np.zeros((len(model_list_i)))
AIC = np.ones((len(model_list_i)))*100
AIC_vec = np.ones((len(model_list_i)))*100
R2vec = np.zeros((len(model_list_i)))
R2barvec = np.zeros((len(model_list_i)))
SST = np.sum((y_exp-y_exp.mean())**2)  # defining total sum of squares

def conf_curve(ind, ind_exp, param, model, dep_points, res, alpha=0.05):
    # X = points for plotting
    # param = fitted params
    # model is model name
    # assuming standard alpha, i.e. p=0.95
    Ns = ind_exp.size  # sample size
    Nvar = len(param)  # number of regression variables

    # from t distribution
    quant = stats.t.ppf(1.0-alpha/2.0, Ns - Nvar)
    # std deviation of point
    stdn = np.sqrt(1.0/(Ns - Nvar) * np.sum(res**2))

    sx = (ind - ind_exp.mean()) ** 2
    sxdev = np.sum((ind_exp - ind_exp.mean()) ** 2)

    # calculate dx

    dx = quant*stdn*np.sqrt(1.0 + (1.0/Ns) + (sx/sxdev))
    low, upp = dep_points - dx, dep_points + dx
    return low, upp


for model in model_list_i:
    path='./Output/OLS_Inverse/'+model
    model_name = model
    p0 = np.ones(3)  # initial parameter points
    # giving initial guesses
    p0[0] = max(y_exp)*1.4
    uplow = ([max(y_exp), 0, -np.inf], np.inf)
    if (model == 'monomolecular_i' or model == 'HeLegendre_i'or model == 'Korf_i' or model == 'MMF_i' or model == 'Weibull_i' or model == 'MichaelisMenten_i' or model == 'ChapmanRichards_i'):
        uplow = ([max(y_exp), 0, 0], np.inf)

    if (model == 'Power_i'):
        uplow = ([0, 0, -np.inf], [np.inf, np.inf, min(y_exp)])
        p0 = [1, 1, min(y_exp)*0.5]
    if (model == 'Power2_i'):
        uplow = ([0, -np.inf, -np.inf], np.inf)

    result = least_squares(calc_res, p0, method='trf',
                           jac='2-point', max_nfev=100*len(p0), bounds=uplow,loss=loss_fn)

    if not result.success:
        # try again with more accurate jacobian and longer max function evals.
        result = least_squares(calc_res, p0, method='trf', jac='3-point',
                               max_nfev=1000*len(p0)*len(p0), bounds=uplow,loss=loss_fn)

    # Estimating covariance matrix form the jacobian
    # as H = 2*J^TJ and cov = H^{-1}
    # estimating pcov using method from minpack.py
    # Do Moore-Penrose inverse discarding zero singular values.
    _, s, VT = svd(result.jac, full_matrices=False)
    threshold = np.finfo(float).eps * max(result.jac.shape) * s[0] 
    s = s[s > threshold]
    VT = VT[:s.size]
    pcov = np.dot(VT.T / s**2, VT)
    cost = result.cost * 2
    s_sq = cost / (y_exp.size - result.x.size)
    pcov = pcov * s_sq
    perf = (result.cost*2)  # (sum (res^2)*0.5)*2
    
    s2 = perf/y_exp.size

    # computing information criterion
    # leaving out np.log(2*pi)
    logL = -(x_exp.size)/2*(np.log(2*pi)+np.log(s2)+1)
    AIC = 2*len(result.x)-2*logL
    AIC = AIC + (2*len(result.x)*(len(result.x)+1)) / \
        (x_exp.size-len(result.x)-1)
    AIC_vec[i]=AIC
    SSR = sum(calc_res(result.x)**2)  # calculating the SSR

    R2vec[i] = 1 - (SSR)/SST
    R2barvec[i] = 1 - (1-R2vec[i])*(x_exp.size-1) / \
                (x_exp.size-1-len(result.x))
    res = calc_res(result.x)
    res = sum(res**2)    
    i=i+1
    # After indentifying the best AIC the remainder should be scaled: delta AICc = AIC - AICmin.
    # for listing later
    y_points = np.linspace(plot_range[0], plot_range[1], 25)

    if (result.success):
        print('Model: %s 	Residual: %4.2f 	AIC: %4.2f  R2: %4.2f ' %
              (model, result.cost, AIC,R2vec[i-1]))
        certain = True
        x_pred = model_x(result.x, y_exp)
        res = calc_res(result.x)
        a, b, c = unc.correlated_values(result.x, pcov)
        print('Uncertainty')
        print('alpha: ' + str(a))
        print('k: ' + str(b))
        print('I: ' + str(c))

        certain = False
        # calculating the uncertain x values

        x_unc = model_x(np.array([a, b, c]), y_points)
        x_points = unp.nominal_values(x_unc)
        std = unp.std_devs(x_unc)

        certain = True
        low, upp = conf_curve(y_points, y_exp, result.x,
                              model, x_points, res, alpha=0.05)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(x_points, y_points, label='Model')
        ax.plot(low, y_points, 'k--', label='95% Prediction band')
        ax.plot(upp, y_points, 'k--')
        ax.plot(x_exp, y_exp, 'bo', label='Experimental points')

        # uncertainty lines (95% confidence)
        ax.plot(x_points - 1.96*std, y_points,
                c='orange', label='95% Confidence Region')

        ax.plot(x_points + 1.96*std, y_points, c='orange')
        ax.set_xlim([0, None])
        ax.set_ylim([0, None])
        ax.legend(loc='best')
        fig.savefig(path+'.png')
        plt.close(fig)

        # printing out
        file_out = open(path+'_results.txt', 'w')
        f = file_out
        d=''
        e=''
        ff=''
        file_out.write("Parameters: A, B & C	\n\n")
        file_out.write('a: ' + str(a))
        file_out.write('\nb: ' + str(b))
        file_out.write('\nc: ' + str(c))
        file_out.write('\nd: ' + str(d))
        if len(result.x) == 6:
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
        print('Model: %s 	FAILED. Residual: %d 	AIC: %d ' %
              (model, result.cost, AIC))

# Final organising for outputs
# delta AIC is more useful than AIC
AIC_vec = AIC_vec-min(AIC_vec)

path='./Output/Summary_OLS_Inverse.txt'

file_out = open(path, 'w')
file_out.write('Summary of outputs\n')
file_out.write('Input:	%s\n' % file_name)
file_out.write('Loss Method:	%s\n\n' % loss_fn)

file_out.write('Model results in order of increasing deltaAIC:\n')
sorted_list = [model_list_i for _,
               model_list_i in sorted(zip(AIC_vec, model_list_i))]
sorted_AIC = [AIC_vec for _, AIC_vec in sorted(zip(AIC_vec, AIC_vec))]
sorted_R2 = [R2vec for _, R2vec in sorted(zip(AIC_vec, R2vec))]
sorted_R2bar = [R2barvec for _, R2barvec in sorted(zip(AIC_vec, R2barvec))]

file_out.write("%-30s	%4s	%4s	%4s\n" % ('Model', 'AIC', 'R2', 'R2bar'))

for i in range(len(sorted_list)):
    file_out.write("%-30s	%4.4f	%4.4f	%4.4f\n" %
                   (sorted_list[i], sorted_AIC[i], sorted_R2[i], sorted_R2bar[i]))

print('Regression with OLS_inverse, with  %5s loss function has finished. Best model is %-30s.'%(loss_fn,sorted_list[0]))
print('\nSee %s for more detials.' %(path))