import numpy as np

def mse(ytrue, ypred):
    #Mean Squared Error
    try:
        # ensure that ytrue and ypred are np.arrays
        #*REF: isinstance suggested by ChatGPT
        #*REF: https://github.com/llSourcell/loss_functions_explained/blob/master/Loss%20Functions%20in%20Machine%20learning%20.ipynb

        if isinstance(ytrue, np.ndarray) and isinstance(ypred, np.ndarray):
            n = ytrue.shape[0]
            error = ytrue - ypred
            return (1 / n) * np.sum(np.square(error))
        else:
            print('Let ypred and ytrue be numpy arrays')
    except Exception as e:
        print(f"An error occurred: {e}")


def mae(ytrue, ypred):
    #Mean Absolute Error
    try: 
        #*REF: https://github.com/llSourcell/loss_functions_explained/blob/master/Loss%20Functions%20in%20Machine%20learning%20.ipynb
        if isinstance(ytrue, np.array) and isinstance(ypred, np.array):
            n = ytrue.shape[0]
            error = ytrue - ypred
            return (1 / n) * np.sum(np.abs(error))
        else:
            print('Let ypred and ytrue be numpy arrays')
    except Exception as e:
        print(f'An error occured: {e}')

def crossEntropy(ytrue, ypred):
    #Cross Entropy Loss
        #*REF: https://youtu.be/EJRFP3WmS6Q?si=arJO-cPk_ECTCGWf
        #*REF: https://github.com/Tunjii10/Neural-Network-from-Srcatch/blob/main/src/Activation_Loss_Functions/activations.py
        
        #ensure ytrue and ypred are the same shape
        assert ytrue.shape == ypred.shape, 'Shape of ytrue and ypred must match' #This line suggested by ChatGPT

        ypred = np.clip(ypred, 1e-15, 1-1e-15) #clip to avoid exp overflow
        n = ytrue.shape[0]
        loss = -1 * ((ytrue * np.log(ypred)) + (1 - ytrue) * np.log(1 - ypred))
        return np.sum(loss) / n