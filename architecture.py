import numpy as np

# this is the architecture of the neural network
def architecture():
  try:
    while True:
        # collect input from the user
      print('WELCOME!!\n --------------------------')
      layers()
      print('\n --------------------')
      epoch()
      print('\n --------------------')
      learning_rate()
      print('\n --------------------')
      batch_size()
      print('\n --------------------')
      loss_function()
      print('\n --------------------')
      gradient_descent()
      break
  except:
    print('Invalid input! Please try again')
    architecture()

def layers():
  try:
    # defining the number of hidden layers and nodes
    while True:
      hidden_layers = eval(input('How many hidden layers do you want in your network? '))
      for i in range(hidden_layers):
        hidden_layer_nodes = eval(input(f'How many nodes do you want in the {i + 1}st hidden layer?'))

        #define the activation function for each layer
        print('What activation function do you want in this layer?')
        print('1. Sigmoid')
        print('2. Tanh')
        print('3. ReLU')
        print('4. Linear')
        print('5. Softmax')

        activation_choice = eval(input('Enter your choice(1-5): '))
        print('_______________\n')
        activation_function_list = []

        #store the choice of activation function in a list
        activation_function_list.append(activation_choice)

      #defining the number of output nodes
      output_layer_nodes = eval(input('\nHow many nodes do you want in the output layer? '))
      activation_choice = eval(input('What activation function do you want in the output layer? '))

      #idea: output activation function would be activation_function_list[-1]
      activation_function_list.append(activation_choice)
      break

  except:
    print('Invalid input! Please try again')
    layers()

def epoch():
  try:
    while True:
      epochs = eval(input('How many epochs do you want to train your network?'))
      break
  except:
    print('Invalid input! Please try again')
    epoch()

def learning_rate():
  try:
    while True:

      learning_rate_list = []
      list_of_type_of_learning_rate = []
      
      print('What type of learning rate do you want to use?: ')
      print('1. Constant')
      print('2. TBD')

      type_of_learning_rate = eval(input('Enter your choice(1-2): '))
      list_of_type_of_learning_rate.append(type_of_learning_rate)

      if type_of_learning_rate == 1:
        learning_rate = eval(input('What learning rate do you want to use? '))
        learning_rate_list.append(learning_rate)
      else:
        learning_rate = eval(input('What learning rate do you want to use? '))
        learning_rate_list.append(learning_rate)

      break
  except:
    print('Invalid input! Please try again')
    learning_rate()

def batch_size():
  try:
    while True:
      batch_size = eval(input('What batch size do you want to use? '))
      break
  except:
    print('Invalid input! Please try again')
    batch_size()

def loss_function():
  try:
    while True:
      print('What loss function do you want to use?')
      print('1. Binary Cross Entropy')
      print('2. Mean Squared Error')
      print('3. Mean Absolute Error')

      loss_choice = eval(input('Enter your choice(1-3): '))
      loss_function_list = []
      loss_function_list.append(loss_choice)
      break
  except:
    print('Invalid input! Please try again')
    loss_function()

def gradient_descent():
  try:
    while True:
      print('What type of gradient descent do you want to use?')
      print('1. Stochastic Gradient Descent')
      print('2. Batch Gradient Descent')
      print('3. Sequential Gradient Descent')

      gradient_choice = eval(input('Enter your choice(1-3): '))
      gradient_descent_list = []
      gradient_descent_list.append(gradient_choice)
      break
  except:
    print('Invalid input! Please try again')
    gradient_descent()

def run():
  architecture()

run()

#References:
# Code structure idea: https://github.com/pineappleflavour/Inventory-management-system/tree/main