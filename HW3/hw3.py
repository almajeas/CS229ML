import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Generate Nonlinear samples y =  x1 + x2**2 + (x1 - cos(x2))**2
def nonlinearfunction(x):
    output = []
    output.append(x[:,0] + x[:,1]**2 + (x[:,0]  - np.cos(x[:,1]))**2 + np.random.normal(0, 0.1, len(x)))
    return np.array(output).T

# sigmoid activation function
def sigmoid(x):
    return  1.0 / (1.0 + np.exp(-1.0 * x))

#sigmoid derivative function
def sigmoidDerivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

#ReLU Activation Function
def ReLU(x):
    return x * (x > 0)

#ReLU Derivative function
def ReLUDerivative(x):
    x[x<=0] = 0
    x[x>0] = 1
    return x

#Activation function in Use
def activation(x):
    return sigmoid(x)
    # return ReLU(x)

#derivative function in use
def activationDerivative(x):
    return sigmoidDerivative(x)
    # return ReLUDerivative(x)

#Root Mean Square Error function
def RMSE(y, t):
    return np.sqrt(np.sum((y - t.T)**2))/ len(y)

epochs = 20 # number of epochs
learning_rate = 0.0001 # learning rate
input_dimensions = 2 #Input variables for the NN first layer
input_points = 1250 # total generated samples
output_dimensions = 1 #output of the NN layer

# Generate re()
# BGD_Adadelta()andom samples + noise
X = np.random.uniform(low=0, high=1, size=(input_points, input_dimensions))

# Calculate Ground Truth for the generated samples using non-linear function
T = nonlinearfunction(X)

#Split into Training and Test Sets
x_train, x_test, t_train, t_test = train_test_split(X, T, test_size=0.20, random_state=42)

# Define Neural Network

## structure of NN in array from, each element is a layer.
## starting with input layer and ending with output layer
neurons_per_layer = np.array([input_dimensions, 30,  output_dimensions])


NN = list() ##Neural Network object
W = list() ## Weights
Wgrad = list() ## Weights gradient calculations (back propagation)
B = list() ## Biases
Bgrad = list() ## Biases gradient calculations (back propagation)
Wgrads = list()
Bgrads = list()
Eg2W = list()
Eg2B = list()
## Initialize Weights, Biases, Gradients
def initialize():
    ##Initialize NN
    NN.clear() ##Neural Network object
    W.clear() ## Weights
    Wgrad.clear() ## Weights gradient calculations (back propagation)
    B.clear() ## Biases
    Bgrad.clear() ## Biases gradient calculations (back propagation)
    Wgrads.clear()
    Bgrads.clear()
    Eg2W.clear()
    Eg2B.clear()
    for i in range(len(neurons_per_layer)):
        NN.append(np.zeros((neurons_per_layer[0], 1)))
    for i in range(len(neurons_per_layer) -1 ):
        W.append(np.random.randn(neurons_per_layer[i+1], neurons_per_layer[i]))
        Wgrad.append(np.zeros((neurons_per_layer[i+1], neurons_per_layer[i])))
        Wgrads.append(np.zeros((neurons_per_layer[i+1], neurons_per_layer[i])))
        Eg2W.append(np.zeros((neurons_per_layer[i+1], neurons_per_layer[i])))
        B.append(np.random.randn(neurons_per_layer[i+1] , 1))
        Bgrad.append(np.zeros((neurons_per_layer[i+1] , 1)))
        Bgrads.append(np.zeros((neurons_per_layer[i+1] , 1)))
        Eg2B.append(np.zeros((neurons_per_layer[i+1] , 1)))


## forward pass 
def forward():
    for i in range(len(W)):
        NN[i+1] = np.dot(W[i], NN[i]) + B[i]
## back propagation (l is the current layer, seg is the error, Z is the input)
def backpropagate(l,  seg, Z):
    if l < 0:
        return   
    Wgrad[l] = np.dot(seg,  Z.T)
    Bgrad[l] = np.sum(seg, axis=1, keepdims=True)/len(x_train)
    backpropagate(l-1, activationDerivative(NN[l])  * np.dot(W[l].T, seg), activation(NN[l-1]) )
    return


## Neural Network training. BGD
def BGD():
    ##lists to keep track of errors for plotting.
    training_errors = []
    testing_errors = []
    for epoch in range(epochs):
        NN[0] = x_train.T
        forward()
        backpropagate(len(Wgrad) -1 , np.subtract(NN[-1], t_train.T), activation(NN[-2]))
        for l in range(len(W)):
            W[l] = W[l] - learning_rate * Wgrad[l]
            B[l] = B[l] - learning_rate * Bgrad[l]
        training_errors.append(RMSE(NN[-1],t_train))
        NN[0] = x_test.T 
        forward()
        testing_errors.append(RMSE(NN[-1], t_test))
        print(f"{epoch} - {training_errors[-1]}")

    # # plotting for HW
    # plt.title(f"HW3.2.1 RMSE with epochs on Training Set\nNeural Network Layers: {neurons_per_layer}")
    # plt.plot(training_errors, color='orange', label='Training Set')
    # plt.xlabel("Epochs")
    # plt.ylabel("Error")
    # plt.legend()
    # plt.show()

    # plt.title(f"HW3.2.2 RMSE with epochs on Training and Testing Sets\nNeural Network Layers: {neurons_per_layer}")
    # plt.plot(training_errors, color='orange', label='Training Set')
    # plt.plot(testing_errors, color='green', label="Testing Set")
    # plt.xlabel("Epochs")
    # plt.ylabel("Error")
    # plt.legend()
    # plt.show()


    plt.title(f"HW3.Bonus Baseline BGD RMSE with epochs on Training and Testing Sets\nNeural Network Layers: {neurons_per_layer}")
    plt.plot(training_errors, color='orange', label=f'BGD Training Set')
    plt.plot(testing_errors, color='green', label=f"BGD Testing Set")
    plt.xlabel("Epochs")
    plt.ylabel("Error")
    plt.legend()
    # plt.show()




print("=================================")
def BGD_AdaGrad():
    epsilon = 10e-8
    learning_rate = 1
    ##lists to keep track of errors for plotting.
    training_errors = []
    testing_errors = []
    for epoch in range(epochs):
        NN[0] = x_train.T
        forward()
        backpropagate(len(Wgrad) -1 , np.subtract(NN[-1], t_train.T), activation(NN[-2]))
        training_errors.append(RMSE(NN[-1],t_train))
        for l in range(len(W)):
            Wgrads[l] += Wgrad[l]**2
            Bgrads[l] += Bgrad[l]**2
            W[l] = W[l] - ((learning_rate/ np.sqrt(epsilon + Wgrads[l])) * Wgrad[l])
            B[l] = B[l] - ((learning_rate/ np.sqrt(epsilon + Bgrads[l])) * Bgrad[l])
        NN[0] = x_test.T 
        forward()
        testing_errors.append(RMSE(NN[-1], t_test))
        print(f"{epoch} - {training_errors[-1]}")

    plt.title(f"HW3.Bonus AdaGrad RMSE with epochs on Training and Testing Sets\nNeural Network Layers: {neurons_per_layer}")
    plt.plot(training_errors, color='red', label='AdaGrad Training Set')
    plt.plot(testing_errors, color='blue', label="AdaGrad Testing Set")
    plt.xlabel("Epochs")
    plt.ylabel("Error")
    plt.legend()
    # plt.show()



def BGD_RMSprop():
    gamma = 0.9
    epsilon = 10e-8
    # learning_rate = 1
    ##lists to keep track of errors for plotting.
    training_errors = []
    testing_errors = []
    for epoch in range(epochs):
        NN[0] = x_train.T
        forward()
        backpropagate(len(Wgrad) -1 , np.subtract(NN[-1], t_train.T), activation(NN[-2]))
        training_errors.append(RMSE(NN[-1],t_train))
        for l in range(len(W)):
            Eg2W[l] = (1 - gamma)*Wgrad[l]**2 + gamma * Eg2W[l]
            Eg2W[l] = (1 - gamma)*Wgrad[l]**2 + gamma * Eg2B[l]
            W[l] = W[l] - ((learning_rate/ np.sqrt(epsilon + Eg2W[l])) * Wgrad[l])
            B[l] = B[l] - ((learning_rate/ np.sqrt(epsilon + Eg2B[l])) * Bgrad[l])
        NN[0] = x_test.T 
        forward()
        testing_errors.append(RMSE(NN[-1], t_test))
        print(f"{epoch} - {training_errors[-1]}")

    plt.title(f"HW3.Bonus AdaGrad RMSE with epochs on Training and Testing Sets\nNeural Network Layers: {neurons_per_layer}")
    plt.plot(training_errors, color='magenta', label='RMSprob Training Set')
    plt.plot(testing_errors, color='gray', label="RMSprob Testing Set")
    plt.xlabel("Epochs")
    plt.ylabel("Error")
    plt.legend()
    plt.show()

initialize()
BGD()
initialize()
BGD_AdaGrad()
initialize()
BGD_RMSprop()