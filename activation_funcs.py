import numpy as np

def sigmoid(x):
    #*REF: https://github.com/lionelmessi6410/Neural-Networks-from-Scratch/blob/main/model.py
    #forward path
    forward_sigmoid = 1 / (1 + np.exp(-x))

    #backward path
    backward_sigmoid = np.exp(-x) / (np.exp(-x) + 1) ** 2

    return forward_sigmoid, backward_sigmoid

def tanh(x):
  #*REF: https://github.com/Tunjii10/Neural-Network-from-Srcatch/blob/main/src/Activation_Loss_Functions/activations.py

  #forward path
  forward_tanh = (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
  #*REF: Lecture notes

  #backward path
  backward_tanh = 1 - forward_tanh ** 2

  return forward_tanh, backward_tanh

def relu(x):
    #forward path
    forward_relu = np.maximum(0, x)

    #backward path
    backward_relu = np.where(x > 0, 1, 0)

    return forward_relu, backward_relu

def linear(x):
    #*REF: Lecture notes
    #forward path
    forward_linear = x

    #backward path
    backward_linear = np.ones_like(x) # differentiates to a constant

    return forward_linear, backward_linear

def softmax(x):
    #REF*: https://youtu.be/8ah-qhvaQqU?si=2m9d5VgWZ2Auj3ss
    #REF*: https://github.com/lionelmessi6410/Neural-Networks-from-Scratch/blob/main/model.py
    #REF*: keepdims = True recommended by Google Gemini.

    return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)
    #additional code: to use softmax, the no. of output nodes should be the same as the no. of ylabels in dataset