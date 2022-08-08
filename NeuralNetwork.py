import pandas as pd
import numpy as np
from scipy.special import expit
import copy

class MLP(object):

    #Initialize all the class variable from the input recieved
    def __init__(self, learning_rate, activation_function, hidden_layers, optimizer, batch_size, epochs, seed, beta = 0.9, beta2 = 0.9):
        self.learning_rate = learning_rate
        self.activation_function = activation_function
        self.hidden_layers = hidden_layers
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.epochs = epochs
        self.seed = seed
        self.loss_per_epoch = []
        self.eps = 1e-8
        self.beta = beta
        self.beta2 = beta2




    # Initialize weights randomly based on hidden layer and output neurons. 
    #Intialize momentum and Gi array as required depending upon optimizer.
    def intilizate_weights_randomly(self, layers):
        np.random.seed(self.seed)
        self.parameter = {}
        self.parameter["weight"] = {}
        self.parameter["bias"] = {}
        self.parameter["z"] = {}
        self.parameter["activation"] = {}
        self.parameter["der_a"] = {}
        self.parameter["der_z"] = {}
        self.parameter["der_w"] = {}
        self.parameter["der_b"] = {}

        if self.optimizer == 'gradient_descent_with_momentum' or self.optimizer == "adam" or self.optimizer == "nag":
            self.parameter["momentum"] = {} 
            self.parameter["bias_momentum"] = {}
        
        if self.optimizer == "adagrad" or self.optimizer == 'rmsprop' or self.optimizer == "adam":
            self.parameter["gi_w"] = {}
            self.parameter["gi_b"] = {}

        for i in range(len(layers) - 1):
            self.parameter["weight"][i+1] = np.random.randn(layers[i+1], layers[i])
            # self.parameter["bias"][i+1] = np.random.randn(layers[i+1], 1)
            self.parameter["bias"][i+1] = np.zeros((layers[i+1], 1))
            if self.optimizer == 'gradient_descent_with_momentum' or self.optimizer == "adam" or self.optimizer == "nag":
                self.parameter["momentum"][i+1] = np.zeros((layers[i+1], layers[i]))
                self.parameter["bias_momentum"][i+1] =  np.zeros((layers[i+1], 1))
            
            if self.optimizer == 'adagrad' or self.optimizer == 'rmsprop' or self.optimizer == "adam":
                self.parameter["gi_w"][i+1] = np.zeros((layers[i+1], layers[i]))
                self.parameter["gi_b"][i+1] =  np.zeros((layers[i+1], 1))
            
    #While saving the model, deleting useless information to reduce file size.
    def empty_variables(self):
        self.parameter["z"] = {}
        self.parameter["activation"] = {}
        self.parameter["der_a"] = {}
        self.parameter["der_z"] = {}
        self.parameter["der_w"] = {}
        self.parameter["der_b"] = {}
        if self.optimizer == "nag":
            self.nag = {}

    #Depending upon data points, hidden layer and output classes, create layer sizes.
    def create_layers(self, X, Y):
        self.layers = []
        self.layers.append(X.shape[1])
        self.layers.extend(self.hidden_layers)
        no_of_classes = np.unique(Y).shape[0]
        if no_of_classes == 2:
            self.layers.append(1)
        else:
            self.layers.append(no_of_classes)

    #create virtual output
    def create_output_for_each_layer(self, Y):
        self.output = np.zeros((Y.shape[0], self.layers[-1]))
        for idx,value in enumerate(Y):
            self.output[idx][value[0]] = 1

    #Calculate activation value
    def activation(self, z, act_type):
        if act_type == 'sigmoid':
            return expit(z)
        elif act_type == 'tanh':
            return np.tanh(z)
        elif act_type == 'relu':
            return np.maximum(z, 0 + self.eps)

    #Calculate softmax value
    def softmax(self, z):
        temp = np.exp(z - np.max(z, axis = 0))
        return temp / temp.sum(axis = 0, keepdims = True)

    #Calculate gradient of all activation function
    def activation_gradient(self, z, act_type):
        if act_type == 'sigmoid':
            return expit(z) * (1 - expit(z))
        elif act_type == 'tanh':
            return 1-np.tanh(z)**2
        elif act_type == 'relu':
            der = copy.deepcopy(z)
            der[der <= 0] = 0 + self.eps
            der[der > 0] = 1
            return der
                

        

    # Call diffenrent fit function depending upon optimizers
    def fit(self, X, Y):
        self.create_layers(X, Y)
        self.intilizate_weights_randomly(self.layers)
        self.create_output_for_each_layer(Y)
        if self.optimizer == 'gradient_descent':
            self.fit_gradient_descent(X, Y)
        elif self.optimizer == 'gradient_descent_with_momentum':
            self.fit_gradient_descent_with_momentum(X, Y)
        elif self.optimizer == 'nag':
            self.fit_nag(X, Y)
        elif self.optimizer == 'adagrad':
            self.fit_adagrad(X, Y)
        elif self.optimizer == 'rmsprop':
            self.fit_rmsprop(X, Y)
        elif self.optimizer == 'adam':
            self.fit_adam(X, Y)

    def fit_nag(self, X, Y):
        batches = int(X.shape[0]/self.batch_size)
        for e in range(self.epochs):
            loss_epoch = 0.0
            for b in range(batches):
                x = X[self.batch_size * b : self.batch_size * b + self.batch_size, ]
                y = Y[self.batch_size * b : self.batch_size * b + self.batch_size, ]
                self.create_output_for_each_layer(y)
                self.forward_propagation(x)
                self.compute_loss()
                self.backpropagation(x)
                self.update_weight_NAG_momentum(x, y)
                loss_epoch = loss_epoch + self.loss
            self.loss_per_epoch.append(loss_epoch/batches)

            print("Loss is: ", loss_epoch/batches," at epoch: ", e, "/", self.epochs)


    def fit_adam(self, X, Y):
        batches = int(X.shape[0]/self.batch_size)
        self.t = 0
        for e in range(self.epochs):
            loss_epoch = 0.0
            for b in range(batches):
                x = X[self.batch_size * b : self.batch_size * b + self.batch_size, ]
                y = Y[self.batch_size * b : self.batch_size * b + self.batch_size, ]
                self.create_output_for_each_layer(y)
                self.forward_propagation(x)
                self.compute_loss()
                self.backpropagation(x)
                self.t = self.t + 1
                self.update_weight_adam(self.t)
                loss_epoch = loss_epoch + self.loss
            self.loss_per_epoch.append(loss_epoch/batches)

            print("Loss is: ", loss_epoch/batches," at epoch: ", e,"/", self.epochs)      

    def fit_rmsprop(self, X, Y):
        batches = int(X.shape[0]/self.batch_size)
        for e in range(self.epochs):
            loss_epoch = 0.0
            for b in range(batches):
                x = X[self.batch_size * b : self.batch_size * b + self.batch_size, ]
                y = Y[self.batch_size * b : self.batch_size * b + self.batch_size, ]
                self.create_output_for_each_layer(y)
                self.forward_propagation(x)
                self.compute_loss()
                self.backpropagation(x)
                self.update_weight_rmsprop()
                loss_epoch = loss_epoch + self.loss
            self.loss_per_epoch.append(loss_epoch/batches)

            print("Loss is: ", loss_epoch/batches," at epoch: ", e,"/", self.epochs)

    def fit_adagrad(self, X, Y):
        batches = int(X.shape[0]/self.batch_size)
        for e in range(self.epochs):
            loss_epoch = 0.0
            for b in range(batches):
                x = X[self.batch_size * b : self.batch_size * b + self.batch_size, ]
                y = Y[self.batch_size * b : self.batch_size * b + self.batch_size, ]
                self.create_output_for_each_layer(y)
                self.forward_propagation(x)
                self.compute_loss()
                self.backpropagation(x)
                self.update_weight_adagrad()
                loss_epoch = loss_epoch + self.loss
            self.loss_per_epoch.append(loss_epoch/batches)

            print("Loss is: ", loss_epoch/batches," at epoch: ", e,"/", self.epochs)
        
    def fit_gradient_descent_with_momentum(self, X, Y):
        batches = int(X.shape[0]/self.batch_size)
        for e in range(self.epochs):
            loss_epoch = 0.0
            for b in range(batches):
                x = X[self.batch_size * b : self.batch_size * b + self.batch_size, ]
                y = Y[self.batch_size * b : self.batch_size * b + self.batch_size, ]
                self.create_output_for_each_layer(y)
                self.forward_propagation(x)
                self.compute_loss()
                self.backpropagation(x)
                self.update_weight_momentum()
                loss_epoch = loss_epoch + self.loss
            self.loss_per_epoch.append(loss_epoch/batches)

            print("Loss is: ", loss_epoch/batches," at epoch: ", e,"/", self.epochs)

    
    def fit_gradient_descent(self, X, Y):
        batches = int(X.shape[0]/self.batch_size)

        for e in range(self.epochs):
            loss_epoch = 0.0
            for b in range(batches):
                x = X[self.batch_size * b : self.batch_size * b + self.batch_size, ]
                y = Y[self.batch_size * b : self.batch_size * b + self.batch_size, ]
                self.create_output_for_each_layer(y)
                self.forward_propagation(x)
                self.compute_loss()
                self.backpropagation(x)
                self.update_weight()
                loss_epoch = loss_epoch + self.loss
            self.loss_per_epoch.append(loss_epoch/batches)

            print("Loss is: ", loss_epoch/batches," at epoch: ", e,"/", self.epochs)
    

    def forward_propagation(self, X):

        #Compute z and activation for first layer

        self.parameter["z"][1] = np.dot(self.parameter["weight"][1], X.T) + self.parameter["bias"][1]
        self.parameter["activation"][1] = self.activation(self.parameter["z"][1], self.activation_function[0])

        #Compute a and activation for middle layers
        for i in range(2, (len(self.hidden_layers) + 1)):

            self.parameter["z"][i] = np.dot(self.parameter["weight"][i], self.parameter["activation"][i-1]) + self.parameter["bias"][i]
            self.parameter["activation"][i] = self.activation(self.parameter["z"][i], self.activation_function[i-1])

        #Compute z and activation for output layer
        self.parameter["z"][len(self.hidden_layers) + 1] = np.dot(self.parameter["weight"][len(self.hidden_layers) + 1], 
                                                                self.parameter["activation"][len(self.hidden_layers)]) + self.parameter["bias"][len(self.hidden_layers) + 1]


        self.parameter["activation"][len(self.hidden_layers) + 1] = self.softmax(self.parameter["z"][len(self.hidden_layers) + 1])

    #Compute cross entropy loss
    def compute_loss(self):
                
        hypothesis = self.parameter["activation"][len(self.hidden_layers) + 1].T
        self.loss =  -np.mean(self.output * np.log(hypothesis + self.eps))

    
    def backpropagation(self, X):
        l = len(self.hidden_layers) + 1

        self.parameter["der_z"][l] = self.parameter["activation"][l] - self.output.T
        self.parameter["der_w"][l] = (np.dot(self.parameter["der_z"][l], self.parameter["activation"][l-1].T))/self.batch_size
        self.parameter["der_b"][l] = ((np.sum(self.parameter["der_z"][l], axis=1, keepdims=True)))/self.batch_size
        self.parameter["der_a"][l-1] = np.dot(self.parameter["weight"][l].T, self.parameter["der_z"][l])

        for i in range (l-1, 0 , -1):
            self.parameter["der_z"][i] = self.parameter["der_a"][i] * self.activation_gradient(self.parameter["z"][i], self.activation_function[i - 1])
            self.parameter["der_b"][i] = ((np.sum(self.parameter["der_z"][i], axis=1, keepdims=True)))/self.batch_size
            if i == 1:
                self.parameter["der_w"][i] = (np.dot(self.parameter["der_z"][i], X))/self.batch_size
            else:
                self.parameter["der_w"][i] = (np.dot(self.parameter["der_z"][i], self.parameter["activation"][i-1].T))/self.batch_size
            if i > 1:
                self.parameter["der_a"][i-1] = np.dot(self.parameter["weight"][i].T, self.parameter["der_z"][i])

    
    def update_weight(self):
        for i in range(1, len(self.hidden_layers) + 1 + 1):
            self.parameter["weight"][i] = self.parameter["weight"][i] - self.learning_rate * self.parameter["der_w"][i]
            self.parameter["bias"][i] = self.parameter["bias"][i] - self.learning_rate * self.parameter["der_b"][i]

    def update_weight_momentum(self):
        for i in range(1, len(self.hidden_layers) + 1 + 1):
            change = (self.beta * self.parameter["momentum"][i]) + ( (1 - self.beta) * self.parameter["der_w"][i])
            self.parameter["weight"][i] = self.parameter["weight"][i] - self.learning_rate * change
            self.parameter["momentum"][i] = change

            change_bias = (self.beta * self.parameter["bias_momentum"][i]) + (self.learning_rate * self.parameter["der_b"][i])
            self.parameter["bias"][i] = self.parameter["bias"][i] - change_bias
            self.parameter["bias_momentum"][i] = change_bias
    
    def update_weight_NAG_momentum(self, X, Y):
        self.gradient_nag(X, Y)
        for i in range(1, len(self.hidden_layers) + 1 + 1):
            change = (self.beta * self.parameter["momentum"][i]) + ( (1 - self.beta ) *  self.nag["der_w"][i])
            self.parameter["weight"][i] = self.parameter["weight"][i] - self.learning_rate * change
            self.parameter["momentum"][i] = change

            change_bias = (self.beta * self.parameter["bias_momentum"][i]) + ( (1 - self.beta ) * self.nag["der_b"][i])
            self.parameter["bias"][i] = self.parameter["bias"][i] - self.learning_rate * change_bias
            self.parameter["bias_momentum"][i] = change_bias

    def gradient_nag(self, X, Y):
        self.nag = {}
        self.nag["weight"] = {}
        self.nag["bias"] = {}
        self.nag["z"] = {}
        self.nag["activation"] = {}
        self.nag["der_a"] = {}
        self.nag["der_z"] = {}
        self.nag["der_w"] = {}
        self.nag["der_b"] = {}
        
        for i in range(1, len(self.hidden_layers) + 1 + 1):
            self.nag["weight"][i] = self.parameter["weight"][i] - self.beta * self.parameter["momentum"][i]
            self.nag["bias"][i] = self.parameter["bias"][i] - self.beta * self.parameter["bias_momentum"][i]

        l = len(self.hidden_layers) + 1
        self.nag["der_z"][l] = self.parameter["activation"][l] - self.output.T
        self.nag["der_w"][l] = (np.dot(self.nag["der_z"][l], self.parameter["activation"][l-1].T))/self.batch_size
        self.nag["der_b"][l] = ((np.sum(self.nag["der_z"][l], axis=1, keepdims=True)))/self.batch_size
        self.nag["der_a"][l-1] = np.dot(self.nag["weight"][l].T, self.nag["der_z"][l])
        for i in range (l-1, 0 , -1):
            self.nag["der_z"][i] = self.nag["der_a"][i] * self.activation_gradient(self.parameter["z"][i], self.activation_function[i - 1])
            self.nag["der_b"][i] = ((np.sum(self.nag["der_z"][i], axis=1, keepdims=True)))/self.batch_size
            if i == 1:
                self.nag["der_w"][i] = (np.dot(self.nag["der_z"][i], X))/self.batch_size
            else:
                self.nag["der_w"][i] = (np.dot(self.nag["der_z"][i], self.parameter["activation"][i-1].T))/self.batch_size
            if i > 1:
                self.nag["der_a"][i-1] = np.dot(self.nag["weight"][i].T, self.nag["der_z"][i])


        




    def update_weight_adagrad(self):
        for i in range(1, len(self.hidden_layers) + 1 + 1):
            self.parameter["gi_w"][i] = self.parameter["gi_w"][i] + np.square(self.parameter["der_w"][i])
            self.parameter["weight"][i] = self.parameter["weight"][i] - (self.learning_rate * self.parameter["der_w"][i])/np.sqrt(self.parameter["gi_w"][i] + self.eps)

            self.parameter["gi_b"][i] = self.parameter["gi_b"][i] + np.square(self.parameter["der_b"][i])
            self.parameter["bias"][i] = self.parameter["bias"][i] - (self.learning_rate * self.parameter["der_b"][i])/np.sqrt(self.parameter["gi_b"][i] + self.eps)

    def update_weight_rmsprop(self):
        for i in range(1, len(self.hidden_layers) + 1 + 1):
            self.parameter["gi_w"][i] = (self.beta * self.parameter["gi_w"][i]) + (1 - self.beta)* np.square(self.parameter["der_w"][i])
            self.parameter["weight"][i] = self.parameter["weight"][i] - (self.learning_rate * self.parameter["der_w"][i])/np.sqrt(self.parameter["gi_w"][i] + self.eps)

            self.parameter["gi_b"][i] = (self.beta * self.parameter["gi_b"][i]) + (1 - self.beta)* np.square(self.parameter["der_b"][i])
            self.parameter["bias"][i] = self.parameter["bias"][i] - (self.learning_rate * self.parameter["der_b"][i])/np.sqrt(self.parameter["gi_b"][i] + self.eps)

    
    def update_weight_adam(self, t):
        for i in range(1, len(self.hidden_layers) + 1 + 1):
            #Calculate Momentum
            self.parameter["momentum"][i] = (self.beta * self.parameter["momentum"][i]) + ((1 - self.beta) * self.parameter["der_w"][i])
            self.parameter["bias_momentum"][i] = (self.beta * self.parameter["bias_momentum"][i]) + ((1 - self.beta) * self.parameter["der_b"][i])

            #Calculate Gi using beta2 for accumalation of gradients
            self.parameter["gi_w"][i] = (self.beta2 * self.parameter["gi_w"][i]) + (1 - self.beta2)* np.square(self.parameter["der_w"][i])
            self.parameter["gi_b"][i] = (self.beta2 * self.parameter["gi_b"][i]) + (1 - self.beta2)* np.square(self.parameter["der_b"][i])

            #Bias Correction
            momentum_correction_weights = self.parameter["momentum"][i] / (1 - self.beta ** t)
            momentum_correction_bias = self.parameter["bias_momentum"][i] / (1 - self.beta ** t)

            gi_correction_weights = self.parameter["gi_w"][i] / (1 - self.beta2 ** t)
            gi_correction_bias = self.parameter["gi_b"][i] / (1 - self.beta2 ** t)

            #Updating weights and biases
            self.parameter["weight"][i] = self.parameter["weight"][i] - (self.learning_rate * momentum_correction_weights) / np.sqrt(gi_correction_weights + self.eps)
            self.parameter["bias"][i] = self.parameter["bias"][i] - (self.learning_rate * momentum_correction_bias) / np.sqrt(gi_correction_bias + self.eps)



    #Predict output values of test set
    def predict(self, X):
        self.forward_propagation(X)
        prediction = np.argmax(self.parameter["activation"][len(self.hidden_layers) + 1] , axis = 0)
        return prediction



        

        
            



