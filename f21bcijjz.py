import numpy as np
import pandas as pd
import random
import seaborn as sns #to visualize the loss
import matplotlib.pyplot as plt
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

np.random.seed(17)

from ucimlrepo import fetch_ucirepo
#* REF: https://pypi.org/project/ucimlrepo/

# Fetch the dataset
concrete_compressive_strength = fetch_ucirepo(id=165)
X = concrete_compressive_strength.data.features
y = concrete_compressive_strength.data.targets
data = concrete_compressive_strength.data

## --- PREPROCESSING ---

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def preprocessing(data, target):

  ''' def processing collects the data, selects the X and the y, converts them to np.arrays,
  and returns X_train, X_test, y_train, y_test'''

  # select the X and the y
  # X = data.drop(target, axis=1)
  # y = data[target]

  X = concrete_compressive_strength.data.features
  y = concrete_compressive_strength.data.targets

  # train, test, split the data
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 17)

  # standardize the data
  # scaler = StandardScaler()
  # X_train = scaler.fit_transform(X_train)
  # X_test = scaler.fit_transform(X_test)
  # y_train = scaler.fit_transform(y_train)
  # y_test = scaler.fit_transform(y_test)


  # convert the arrays into numpy arrays
  X_train = np.array(X_train)
  X_test = np.array(X_test)
  y_train = np.array(y_train)
  y_test = np.array(y_test)

  return X_train, X_test, y_train, y_test


## -- BY PASS --
def bypass():
  # bypass allows configuring the network parameter by hand

  '''
  bypass: by passes UI to let user set parameters manually.
  Note: the output node is always 1 as this is a regression problem.
  user_feature_count must be the number of features in your dataset.
  user_hidden_layer_nodes length is the number of hidden layers.
  user_hidden_layer_nodes list is the number of nodes in each hidden layer.
  activation_function_list is the activation function for each hidden layer PLUS the output layer.
  READ THE PREVIOUS LINE AGAIN.
  '''

  iterations = 100 # edit this (a.k.a no. of epoch)
  user_feature_count = '8' # edit this
  user_hidden_layers_nodes = [4, 4, 4, 4] # edit this
  activation_function_list = ['logistic', 'ReLU', 'tanh', 'logistic', 'ReLU'] # edit this
  # swarmsize = 20 # edit this

  # input for architecture requires a string
  user_hidden_layers_nodes = ', '.join(str(i) for i in user_hidden_layers_nodes)
  activation_function_list = ', '.join(activation_function_list)

  return iterations, user_feature_count, user_hidden_layers_nodes, activation_function_list #, swarmsize


## -- UI --
def UI():
    # User interface
    ''' Interactive User Interface to let the user to enter the network parameters.
    '''

    try:
        print('WELCOME!! \n----------------------------')

        # Initialization
        user_hidden_layers_nodes = []
        activation_function_list = []

        # Collect variables from the user
        iterations = int(input("Enter the number of iterations (e.g., 100): "))
        user_feature_count = input("Enter the number of attributes from your data (e.g., 8): ")

        num_hidden_layers = int(input("Enter the number of hidden layers: "))

        # Collect nodes and activation functions for each hidden layer
        for i in range(num_hidden_layers):
            nodes = int(input(f"Enter the number of nodes for hidden layer {i + 1}: "))
            user_hidden_layers_nodes.append(nodes)

            print("Choose an activation function for this layer:")
            print("1. ReLU")
            print("2. tanh")
            print("3. logistic")

            while True:
                choice = input("Enter your choice (1/2/3): ")
                if choice == "1":
                    activation_function_list.append("ReLU")
                    break
                elif choice == "2":
                    activation_function_list.append("tanh")
                    break
                elif choice == "3":
                    activation_function_list.append("logistic")
                    break
                else:
                    print("Invalid choice. Please enter 1, 2, or 3.")

        # Output layer activation function
        print("Choose an activation function for the output layer:")
        print("1. ReLU")
        print("2. tanh")
        print("3. logistic")

        while True:
            output_choice = input("Enter your choice (1/2/3): ")
            if output_choice == "1":
                activation_function_list.append("ReLU")
                break
            elif output_choice == "2":
                activation_function_list.append("tanh")
                break
            elif output_choice == "3":
                activation_function_list.append("logistic")
                break
            else:
                print("Invalid choice. Please enter 1, 2, or 3.")

        # input for the ann architecture is a string
        user_hidden_layers_nodes = ', '.join(str(i) for i in user_hidden_layers_nodes)
        activation_function_list = ', '.join(activation_function_list)

        # Summary of the user input
        print("\nSummary:")
        print(f"Iterations: {iterations}")
        print(f"Feature Count: {user_feature_count}")
        print(f"Number of Hidden Layers: {num_hidden_layers}")
        print(f"Nodes in Each Layer: {user_hidden_layers_nodes}")
        print(f"Activation Functions: {activation_function_list}")


        # Returning input as structured data
        return iterations, user_feature_count, user_hidden_layers_nodes, activation_function_list

    except ValueError as e:
        print(f"Invalid input. Please enter numbers only. Error: {e}")

# Call the function

## -- ANN ARCHITECTURE --
class ANNArchitecture:
  def __init__(self, input_data, user_feature_count, user_hidden_layers_nodes, activation_function_list):
    self.input_data = input_data #X_train

    self.user_feature_count = user_feature_count #number of attributes from your data e.g. 8

    self.user_hidden_layers_nodes = user_hidden_layers_nodes #number of neurons you wanna have for each hidden layer you to be included in the neural network


    #Design the structure of the ANN
    self.count_layers_nodes = [int(user_feature_count.strip())] + [int(item.strip()) for item in user_hidden_layers_nodes.split(",")] + [1]

    self.hidden_layers_activation = [activ.strip() for activ in activation_function_list.split(",")] #Enter an activation function to be used in each of the hidden layers including the output one.
    self.weights = [] #initialise the weights for each layer
    self.biases = [] #initialise the biases for each layer.

    for i in range(len(self.count_layers_nodes) - 1):
      #* REF: ChatGPT suggested code for initializing the weights
      self.weights.append(np.random.randn(self.count_layers_nodes[i+1], self.count_layers_nodes[i]) * 0.1) #0.1 to make the weight as small as possible
      self.biases.append(np.random.randn(self.count_layers_nodes[i+1], 1) * 0.1) #0.1 to make the weight as small as possible

#Actiavation functions to be employed in the model.
  #sigmoid function
  def logistic_func(self, x):
    return 1 / (1 + np.exp(np.clip(-x, -500, 500))) #clip the sigmoid function to avoid overflow

  #Reactified Linear Unit
  def ReLU(self, x):
    return np.maximum(0, x)

  #Hyperbolic tangent
  def Hyperbolic_tan(self, x):
    return np.tanh(x)

  def apply_activation_func(self, x, activ_func):
    if activ_func == 'logistic':
      return self.logistic_func(x)

    if activ_func == 'ReLU':
      return self.ReLU(x)

    if activ_func == 'tanh':
      return self.Hyperbolic_tan(x)

    print("invalid activation function entered.")

  def forward_propagation(self):

    X = self.input_data.T
    for i in range(len(self.weights)):
      Z = np.dot(self.weights[i], X) + self.biases[i]
      X = self.apply_activation_func(Z, self.hidden_layers_activation[i])
    return X

  def loss_func(self, y_true, y_pred):
    '''the loss function used is the mean absolute error.
    '''
    return np.mean(np.abs(y_true - y_pred))
  
## -- PSO --
class PSO:
  #* REF: Essentials of Metaheuristics: https://cs.gmu.edu/~sean/book/metaheuristics/Essentials.pdf

  def __init__(self,ANNArchitecture, swarmsize):
    self.ANNArchitecture = ANNArchitecture
    self.swarmsize = swarmsize
    self.position = [] #container for particles' position
    self.velocity = [] #container for particles' velocity
    self.particle_best_position = []
    self.particle_best_fitness = []
    self.swarm_best_position = None
    self.swarm_best_fitness = float('inf')
    self.position_history = [] # Reset position history


  def particle_structure(self):
      particle_ann = []
      for w in range(len(self.ANNArchitecture.count_layers_nodes)-1):
        particle_ann.append(np.random.randn(self.ANNArchitecture.count_layers_nodes[w+1], self.ANNArchitecture.count_layers_nodes[w]))
      for b in range(len(self.ANNArchitecture.count_layers_nodes)-1):
        particle_ann.append(np.random.randn(self.ANNArchitecture.count_layers_nodes[b+1], 1))
      return particle_ann

  def init_swarm_pos_vel(self):
    for i in range(self.swarmsize):
      self.position.append(self.particle_structure())
      self.velocity.append(self.particle_structure())
      self.particle_best_position.append(self.position[i])
      self.particle_best_fitness.append(float('inf'))

  def fp_particle_structure(self, particle_ann):
    X = self.ANNArchitecture.input_data.T #transpose input data HERE.
    for i in range(len(particle_ann)//2):
      Z = np.dot(particle_ann[i], X) + particle_ann[i+len(particle_ann)//2]
      X = self.ANNArchitecture.apply_activation_func(Z, self.ANNArchitecture.hidden_layers_activation[i])
    return X

  def update_pos_and_vel(self, max_iter, y_train):
    self.position_history.clear()

    for iter in range(max_iter):
      for i in range(self.swarmsize):
        #find a way to propagate using the new particle structure.
        y_pred = self.fp_particle_structure(self.position[i])
        fitness = self.ANNArchitecture.loss_func(y_train, y_pred)
        print(f'Fitness: {fitness}')

        if fitness < self.particle_best_fitness[i]:
          self.particle_best_position[i] = self.position[i]
          self.particle_best_fitness[i] = fitness
        if fitness < self.swarm_best_fitness:
          self.swarm_best_position = self.position[i]
          self.swarm_best_fitness = fitness
          print(f'Swarm best fitness: {self.swarm_best_fitness}')


      for i in range(self.swarmsize):
        #previous fittest location particle
        ppfl = self.position[i]
        #previous fittest location of informants
        pfli = self.particle_best_position[i]
        #previous fittest location of any particle
        spfl = self.swarm_best_position
        for j in range(len(self.position[i])):

          #b = random.random()
          #c = random.random()
          #d = random.random()
          #self.velocity[i][j] = b * self.velocity[i][j] + c * (pfli[j] - ppfl[j]) + d * (spfl[j] - ppfl[j])


          #*REF: Code correction suggested by claude-3.5-sonnet
          # Velocity update too aggressive and not diverse
          w = 0.729 #inertia weight
          c1 = 1.49445 #cognitive constant
          c2 = 1.49445 #social constant
          r1, r2 = random.random(), random.random()
          self.velocity[i][j] = w * self.velocity[i][j] + c1 * r1 * (pfli[j] - ppfl[j]) + c2 * r2 * (spfl[j] - ppfl[j])

      for i in range(len(self.position)):
        for j in range(len(self.position[i])):
          #self.position[i][j] = self.position[i][j] + 0.25 * self.velocity[i][j]

          #*REF: Code correction suggested by claude-3.5-sonnet
          #Consider adding velocity clamping and removing hardcoded learning rate

          v_max = 0.90
          v_min = -0.90
          self.velocity[i][j] = np.clip(self.velocity[i][j], v_min, v_max)
          self.position[i][j] = self.position[i][j] + self.velocity[i][j]

      #return self.swarm_best_position, self.swarm_best_fitness # early convergence mistake suggested by Claude
    
        #**** perf: performance improvement - include position tracking
        positions_flattened = []
        for pos in self.position:
            flat = np.concatenate([p.flatten() for p in pos])
            positions_flattened.append(flat)
        self.position_history.append(positions_flattened)
        #**** end of performance improvement

    return self.swarm_best_position, self.swarm_best_fitness

  # optimizing the weights with the best position from the pso
  def optimize(self, swarm_best_position):
    for i in range(len(swarm_best_position)//2):
      self.ANNArchitecture.weights[i] = swarm_best_position[i]
      self.ANNArchitecture.biases[i] = swarm_best_position[i+len(swarm_best_position)//2]


## -- PCA Trajectory Function --
def get_pca_trajectory(position_history):
    if not position_history:
        return []
    
    all_positions = np.concatenate(position_history, axis=0)
    pca = PCA(n_components=2)
    all_transformed = pca.fit_transform(all_positions)

    # Break into iterations
    n_particles = len(position_history[0])
    iterations_2d = []
    for i in range(len(position_history)):
        start = i * n_particles
        end = (i + 1) * n_particles
        iterations_2d.append(all_transformed[start:end])
    return iterations_2d


## -- MAIN --
def run():

  # Preprocessing
  X_train, X_test, y_train, y_test = preprocessing(data, target)

  # UI and bypass
  try:
    while True:
      print('Bypass or UI?')
      print('1. Bypass')
      print('2. UI')
      print('3. Exit')
      choice = input('Enter your choice (1/2/3): ')

      if choice == '1':
        iterations, user_feature_count, user_hidden_layers_nodes, activation_function_list = bypass()
        break

      elif choice == '2':
        iterations, user_feature_count, user_hidden_layers_nodes, activation_function_list = UI()
        break

      elif choice == '3':
        print('Exiting the program.')
        break

      else:
        print('Invalid choice. Please enter 1, 2, or 3.')

  except ValueError as e:
    print(f"Invalid input. Please enter numbers only. Error: {e}")

  # ANN
  ann = ANNArchitecture(X_train, user_feature_count, user_hidden_layers_nodes, activation_function_list)
  y_pred = ann.forward_propagation()

  # pso
  swarmsize = int(input('Enter swarmsize'))
  pso = PSO(ann, swarmsize)
  pso.init_swarm_pos_vel()
  best_position, best_fitness = pso.update_pos_and_vel(iterations, y_train)

  # optimize with the best position
  pso.optimize(best_position)
  y_train_pred = ann.forward_propagation() #get prediction on the training set

  # test set
  ann.input_data = X_test # temporal switch to test data
  y_test_pred = ann.forward_propagation() #get prediction on the test set
  ann.input_data = X_train # switch back to training data

  #calculate loss function
  train_loss = ann.loss_func(y_train, y_pred)
  test_loss = ann.loss_func(y_test, y_test_pred)

  print(f'Training loss: {train_loss}')
  print(f'Test loss: {test_loss}')

  return y_train_pred, y_test_pred, best_position, best_fitness, train_loss, test_loss,ann


# Run the code
y_train_pred, y_test_pred, best_position, best_fitness, train_loss, test_loss,ann = run()

## -- VISUALIZATION --
# Track losses over iterations
iterations = list(range(10, 151, 10))
swarm_size = 50
train_losses = []
test_losses = []

for i in iterations:
    # Initialize and train
    pso = PSO(ann, swarm_size)
    pso.init_swarm_pos_vel()
    best_position, best_fitness = pso.update_pos_and_vel(i, y_train)

    # Optimize weights with best position
    pso.optimize(best_position)

    # Get predictions
    y_train_pred = ann.forward_propagation()

    # Switch to test data temporarily
    ann.input_data = X_test
    y_test_pred = ann.forward_propagation()
    ann.input_data = X_train  # Switch back

    # Calculate losses
    train_loss = ann.loss_func(y_train, y_train_pred)
    test_loss = ann.loss_func(y_test, y_test_pred)

    train_losses.append(train_loss)
    test_losses.append(test_loss)

# Plot both losses
plt.figure(figsize=(10, 6))
plt.plot(iterations, train_losses, label='Training Loss', marker='o')
plt.plot(iterations, test_losses, label='Test Loss', marker='s')
plt.xlabel('Number of Iterations')
plt.ylabel('Loss')
plt.title('Training and Test Loss vs Iterations')
plt.legend()
plt.grid(True)
plt.show()

# Plot graph of the fitness function against swarmsize
iterations = 100  # Fixed number of iterations
swarm_sizes = list(range(10, 151, 10))  # Test swarm sizes from 10 to 150 in steps of 10
swarm_best_fitness = []

for size in swarm_sizes:
    pso = PSO(ann, size)
    pso.init_swarm_pos_vel()
    best_position, best_fitness = pso.update_pos_and_vel(iterations, y_train)
    swarm_best_fitness.append(best_fitness)

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(swarm_sizes, swarm_best_fitness, marker='o', linestyle='-')
plt.xlabel('Swarm Size')
plt.ylabel('Best Fitness')
plt.title('Swarm Size vs Best Fitness')
plt.grid(True)
plt.show()