"""	Growth Curves
    Copyright 2020 Evren Mert Turan

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""

# IMPORTANT NOTE!
# Errors will occur in this code if one of the fitted parameters wraps around 0.
# e.g. if a = 0.5 +/- 0.6
# then the quoted confidence is incorrect

import matplotlib.pyplot as plt
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

file_name = "Example_Goats_2.txt"
file_name = 'Example_Kangaroo_Finv.txt'
model_list = ['Gompertz_3a_i', 'monomolecular_i', 'vonBertalanffy_i', 'ExtremeValue_i', 'HeLegendre_i', 'Korf_i', 'logistic_i', 'MMF_i', 'Weibull_i', 'MichaelisMenten_i', 'NegativeExponential_i',
              'Power_i', 'Power2_i', 'ChapmanRichards_i']
#
# Open file, load title and read in data
file = open(file_name, "r")
Title = file.readline()
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

shape = data.shape
certain = True

# In general size/mass is more accurate measurement than age, and hence
# should be used as the dependent variable for growth curves

# Common growth models that are described by 3 variables:


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


def ExtremeValue_i(param, y):
    if certain:
        x = -np.log(-((-param[0]+y)/param[0]))
        x = (-param[2]+np.log(x))/param[1]
    else:
        x = -unp.log(-((-param[0]+y)/param[0]))
        x = (-param[2]+unp.log(x))/param[1]
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
SST = np.sum((x_exp-x_exp.mean())**2) # defining total sum of squares
res_vec = np.zeros((len(model_list)))
AIC_vec = np.ones((len(model_list)))*100
R2vec = np.zeros((len(model_list)))

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


for model in model_list:
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
                           jac='2-point', max_nfev=100*len(p0), bounds=uplow)

    if not result.success:
        # try again with more accurate jacobian and longer max function evals.
        result = least_squares(calc_res, p0, method='trf', jac='3-point',
                               max_nfev=1000*len(p0)*len(p0), bounds=uplow)

    # Estimating covariance matrix form the jacobian
    # as H = 2*J^TJ and cov = H^{-1}
    U, s, Vh = svd(result.jac, full_matrices=False)
    threshold = 1e-14  # anything below machine error*100 = 0
    s = s[s > threshold]
    Vh = Vh[:s.size]
    pcov = np.dot(Vh.T / s**2, Vh)
    perf = (result.cost*2)  # (sum (res^2)*0.5)*2
    
    s2 = perf/y_exp.size

    # computing information criterion
    # leaving out np.log(2*pi)
    logL = -(x_exp.size)/2*(np.log(2*pi)+np.log(s2)+1)
    AIC = 2*len(result.x)-2*logL
    AIC = AIC + (2*len(result.x)*(len(result.x)+1)) / \
        (x_exp.size-len(result.x)-1)
    AIC_vec[i]=AIC
    res = calc_res(result.x)
    res = sum(res**2)    
    R2vec[i] = 1- (res)/SST
    i=i+1
    # After indentifying the best AIC the remainder should be scaled: delta AICc = AIC - AICmin.

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
        fig.savefig('./Output/Inverse/'+model+'.png')
        plt.close(fig)
        # printing out
        file_out = open('./Output/Inverse/'+model+'_results.txt', 'w')
        f = file_out
        file_out.write("Parameters: A, B & C	\n\n")
        file_out.write('a: ' + str(a))
        file_out.write('\nb: ' + str(b))
        file_out.write('\nc: ' + str(c))

        file_out.write("\n\nPlotting Data: Predicted \n\n")
         # file_out.close()
        mat = np.matrix([x_points, y_points])
        mat = np.rot90(mat)
           # with open('./Output/'+model+'_results.txt','a') as f:
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


file_out = open('./Output/Inverse/Summary.txt', 'w')
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
