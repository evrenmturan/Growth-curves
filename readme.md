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

## growth-curves_inverse.py

The following 3 parameter functions are fitted:

- Gompertz_3a
- Monomolecular
- von Bertalanffy
* Extreme Value
* He Legendre
* Korf3          
* Logisitic3     
* MMF (Morgan-Mercer-Flodin3a) 
* Weibull (3a)  
* MichaelisMenten
* Negative Exponential
* Power (2 forms)
* Chapman Richards
