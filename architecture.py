# Main architecture function to set up and collect all network parameters
def architecture():
    try:
        network_params = {}

        print('WELCOME!!\n--------------------------')

        # Collect problem type
        network_params["problem_type"] = get_problem_type()

        # Collect layer dimensions and activation functions
        print('\n--------------------')
        network_params["layer_dimensions"] = layers(network_params["problem_type"])

        # Collect epoch count
        print('\n--------------------')
        network_params["epochs"] = epoch()

        # Collect learning rate settings
        print('\n--------------------')
        network_params["learning_rate"] = learning_rate()

        # Collect batch size
        print('\n--------------------')
        network_params["batch_size"] = batch_size()

        # Collect loss function settings
        print('\n--------------------')
        network_params["loss_function"] = loss_function()

        # Collect gradient descent settings
        print('\n--------------------')
        network_params["gradient_descent"] = gradient_descent()

        print("\nNetwork parameters successfully configured:")
        for key, value in network_params.items():
            print(f"{key}: {value}")

        return network_params

    except Exception as e:
        print(f'Error encountered: {e}. Please try again.')

# Collect problem type
def get_problem_type():
    try:
        while True:
            print('What type of problem are you trying to solve: ')
            print('1. Classification')
            print('2. Regression')
            choice = int(input('Enter your choice (1-2): '))
            if choice in [1, 2]:
                return choice
            else:
                print('Invalid choice! Please enter 1 or 2.')
    except ValueError:
        print('Invalid input! Please enter a number.')

# Define layers and activation functions
def layers(problem_type):
    try:
        dimension = {}
        hidden_layer_nodes = []
        activation_function_list = []
        hidden_layers = int(input('How many hidden layers do you want in your network? '))

        for i in range(hidden_layers):
            hidden_layer_nodes_choice = int(input(f'How many nodes do you want in the {i + 1}st hidden layer? '))
            hidden_layer_nodes.append(hidden_layer_nodes_choice)

            print('What activation function do you want in this layer?')
            print('1. Sigmoid')
            print('2. Tanh')
            print('3. ReLU')
            print('4. Linear')
            print('5. Softmax')

            activation_choice = int(input('Enter your choice (1-5): '))
            activation_function_list.append(activation_choice)

        output_activation_choice = int(input('Choose the output layer activation function: '))
        dimension['hidden_layers'] = hidden_layer_nodes
        dimension['hidden_activations'] = activation_function_list
        dimension['output_activation'] = output_activation_choice

        return dimension

    except ValueError:
        print('Invalid input! Please enter a valid number.')

# Collect number of epochs
def epoch():
    try:
        return int(input('How many epochs do you want to train your network? '))
    except ValueError:
        print('Invalid input! Please enter an integer.')

# Collect learning rate settings
def learning_rate():
    try:
        learning_rate_list = []
        print('Choose learning rate type:')
        print('1. Constant')
        print('2. TBD')
        type_of_learning_rate = int(input('Enter learning rate type (1-2): '))
        rate = float(input('Enter the learning rate value: '))
        learning_rate_list.append(rate)
        return type_of_learning_rate, learning_rate_list
    except ValueError:
        print('Invalid input! Please try again.')

# Collect batch size
def batch_size():
    try:
        return int(input('What batch size do you want to use? '))
    except ValueError:
        print('Invalid input! Please enter a positive integer.')

# Collect loss function settings
def loss_function():
    try:
        print('Choose loss function:')
        print('1. Binary Cross Entropy')
        print('2. Mean Squared Error')
        print('3. Mean Absolute Error')
        choice = int(input('Enter your choice (1-3): '))
        return choice
    except ValueError:
        print('Invalid input! Please enter a valid choice.')

# Collect gradient descent settings
def gradient_descent():
    try:
        print('Choose gradient descent type:')
        print('1. Stochastic')
        print('2. Batch')
        print('3. Sequential')
        choice = int(input('Enter your choice (1-3): '))
        return choice
    except ValueError:
        print('Invalid input! Please enter a number.')

# Run the complete setup process
def run():
    architecture()

#References:
# Code structure idea: https://github.com/pineappleflavour/Inventory-management-system/tree/main