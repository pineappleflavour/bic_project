import numpy as np
import pandas as pd
import os

def preprocessing():
  
  ''' def preprocessing collects the data, selects the X and the y, converts them to np.arrays,
  for us to collect the shape of the input and output layer.
  '''

  #get the path of the data, and parse
  path = os.getcwd()
  data = pd.read_csv(path + 'concrete_data.csv')

  #select the X and y
  y = data.iloc[:, -1]
  X = data.iloc[:, :-1]

  #convert the arrays into numpy arrays
  X = np.array(X)
  y = np.array(y)

  #reshape y to be (nXm) matrix for multiplication
  y = y.reshape(len(y), 1)

  return X, y