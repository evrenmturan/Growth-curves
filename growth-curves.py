# Growth Curves
# By Evren Turan

import numpy as np

# Variables

file_name  = "Example_1.txt"
model = "Gompertz"
#
# Open file, load title and read in data
file = open(file_name,"r")
Title = file.readline()
raw_data = file.readlines()

# Double-nested list to get make list of strings to matrix
# the first column must be x data, second column y data
data =np.array([[float(val) for val in line.split()] for line in raw_data[1:]])
shape = data.shape

# Model

# Gompertz
# y = a*exp{-b*exp{-c*x}}
# fits a, b and c
