# Growth-curves
Program to test various growth curves against data.

Regression options include:
	
 - Ordinary Least Squares (OLS)
	 + linear loss
	 + soft_l2 loss
 - Orthogonal Distance Regression (ODR)

These can be specified in Input.txt along with the data file.

## Curves

A wide range of models are provided for fitting. Note that some models can issues in their fit, especially ones with a large number of parameters. If a model is fitted and gives a parameter with a large range, e.g. c = (0 +- 4)e6, this means that this parameter is fitted extremely loosely, and a more reduced form of the model would have been appropriate. Use of such a  model is therefore cautioned.

## Available Models
If number of parameters are not stated in the below, assume 3 parameter form is used.
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
 - Power (2 parameters)
 - Morgan Mercer Flodin
 - Unified Richards (4 parameters)
 - logistic (2 parameters)
 - linear (2 parameters)
 - von Bertalanffy (2 parameters)
 - Richards (4 parameters) : breaks, should be excluded
 - He Legendre (2 parameters)
 - Levakovic (2 parameters)
 - Levakovic
 
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
 - Power (2 parameter)
 - Chapman Richards
 - Extreme Value
 - He Lengendre (2 parameters)
 - Levakovic (2 parameters)  
 - Levakovic
 - Linear (2 parameter)
 
 
