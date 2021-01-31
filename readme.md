# Growth-curves: regression and ranking of various growth curves against experimental data 
This software automatically tests various growth curves against provided experimental data with a selection of different regression options. For more details see the pdf, Growth Curves Report, in the repository. The outputs of the program include:

## Outputs
 - A ranked list of the models, which includes the R2, AIC and BIC of each model. These last two allow for model selection.
 - Plots of each model, including a 95% prediction and confidence band
 - Parameters for each model, including uncertainties in the parameters

## Types of regression
The program can perform three different types of regression:
	
 - Ordinary Least Squares (OLS)
 - Inverted OLS (OLSi)
 - Orthogonal Distance Regression (ODR)

The choice of these options depend on whether the age or growth (y-variable, i.e. size, or mass, etc.) has the largest associated uncertainty/error. Examples of papers on why this is an important decision to make, and the influence of this decision are: [Kaufmann (1981)](https://doi.org/10.1007/BF00347588) and [Myhrvold (2013)](https://doi.org/10.1177/0049124104268644).
OLS should be used if the age estimate is without error, OLSi should be used if the y value (size, mass, etc.) is without error, and ODR should be used if there is error in both variables. More details on ODR are given in the literature folder.

OLS(i) can be used with linear or soft l1 loss. Soft l1 is less sensitive to the presence of outliers and can lead to a more robust regression.


## How to use

Python and the packages: scipy, numpy, uncertainties and matplotlib, are required to run the code. No knowledge of coding is required for use, simply install these required prerequisties, see the [Anaconda documentation](https://docs.anaconda.com/anaconda/install/) for a guide on setting up python.
  
To use the program simply fill in the necessary information in Input.txt file and provide the experimental data as a text file (with . used the decimal separator). See 
"Example_Goats.txt" for how the experimental data should be arranged.

In Input.txt one needs to provide the location of the experimental data, the method of regression, and the number of points used in the plotting. The first line in the file should be the data title, followed by the range desired in the final plots, followed by a blank line and the data in column format. The first column should be age and the second column should be the growth parameter.


## Validation
The code has been compared to various published results. The validation folder contains some comparisons of literature parameter estimates and model choices and ones determined by this code. 

## Available Models

A wide range of models are provided for fitting. Note that some models are (mathematically) equivalent to others but have different names due to being written in a different form

See the file Function List.pdf for the functions with mathematical expressions. Note that some models can experience issues in the fitting fit, and may give errors. E.g. a model with a parameter with very large uncertainty, e.g. c = (0 +- 4)e6. It is typically clear if the fitting worked correctly, especially upon examination of the figures.

The list of models available follows. Note that some models become extremely sensitive when inverted and are not used in the inverted OLS problem. If number of parameters are not stated in the below, then assume this is a 3 parameter form.
### OLS and ODR

 - Gompertz
 - Quadratic
 - Logistic
 - von Bertalanffy
 - Monomolecular
 - Chapman Richards
 - He Legendre
 - Korf
 - Weibull
 - Michaelis Menten
 - Negative Exponential
 - Power
 - Levakovic
 - Morgan Mercer Flodin
 - Unified Richards (4 parameters)
 - Richards (4 parameters) -- regression can easily fail
 - logistic (2 parameters)
 - linear (2 parameters)
 - von Bertalanffy (2 parameters)
 - He Legendre (2 parameters)
 - Levakovic (2 parameters)
 - Power (2 parameters)
 - Extreme Value
 
 
### OLS inverse

 - Gompertz
 - Monomolecular
 - von Bertalanffy
 - He Legendre
 - Korf
 - Logistic
 - Morgan Mercer Flodin
 - Weibull
 - Michaelis Menten
 - Negative Exponential
 - Power
 - Chapman Richards
 - Extreme Value
 - Levakovic
 - He Lengendre (2 parameters)
 - Levakovic (2 parameters)  
 - Power (2 parameter)
 - Linear (2 parameter)
 
 
