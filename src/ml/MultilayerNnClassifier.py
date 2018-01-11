'''
Created on 09 gen 2018

@author: Umberto Griffo

'''

from random import random
from random import seed
from ml.activation.Tanh import Tanh

class MultilayerNnClassifier:
    
    def initialize_network(self, n_inputs, n_hidden, n_outputs):
        '''
        Initialize a new neural network ready for training. 
        It accepts three parameters, the number of inputs, the number of neurons 
        to have in the hidden layer and the number of outputs.
        '''
        network = list()
        # hidden layer has 'n_hidden' neuron with 'n_inputs' input weights plus the bias
        hidden_layer = [{'weights':[random() for i in range(n_inputs + 1)]} for i in range(n_hidden)]
        network.append(hidden_layer)
        output_layer = [{'weights':[random() for i in range(n_hidden + 1)]} for i in range(n_outputs)]
        network.append(output_layer)
        return network
    
    def activate(self, weights, inputs):
        '''
        Calculate neuron activation for an input is the First step of forward propagation
        activation = sum(weight_i * input_i) + bias.
        '''
        activation = weights[-1]  # Bias
        for i in range(len(weights) - 1):
            activation += weights[i] * inputs[i]
        return activation
    
    def forward_propagate(self, network, activation_function, row):
        '''
        Forward propagate input to a network output.
        The function returns the outputs from the last layer also called the output layer.
        '''
        inputs = row
        for layer in network:
            new_inputs = []
            for neuron in layer:
                activation = self.activate(neuron['weights'], inputs)
                neuron['output'] = activation_function.transfer(activation)
                new_inputs.append(neuron['output'])
            inputs = new_inputs
        return inputs
    
    def backward_propagate_error(self, network, activation_function, expected):
        '''
        Backpropagate error and store in neurons.
        
        The error for a given neuron can be calculated as follows:
        
            error = (expected - output) * transfer_derivative(output)
            
        Where expected is the expected output value for the neuron, 
        output is the output value for the neuron and transfer_derivative() 
        calculates the slope of the neuron's output value.
        
        The error signal for a neuron in the hidden layer is calculated as:
        
            error = (weight_k * error_j) * transfer_derivative(output)
            
        Where error_j is the error signal from the jth neuron in the output layer, 
        weight_k is the weight that connects the kth neuron to the current neuron 
        and output is the output for the current neuron.
        '''
        for i in reversed(range(len(network))):
            layer = network[i]
            errors = list()
            if i != len(network) - 1:
                for j in range(len(layer)):
                    error = 0.0
                    for neuron in network[i + 1]:
                        error += (neuron['weights'][j] * neuron['delta'])
                    errors.append(error)
            else:
                for j in range(len(layer)):
                    neuron = layer[j]
                    errors.append(expected[j] - neuron['output'])
            for j in range(len(layer)):
                neuron = layer[j]
                neuron['delta'] = errors[j] * activation_function.transfer_derivative(neuron['output'])
    
    def update_weights(self, network, row, l_rate):
        '''
        Updates the weights for a network given an input row of data, a learning rate 
        and assume that a forward and backward propagation have already been performed.
        
            weight = weight + learning_rate * error * input
            
        Where weight is a given weight, learning_rate is a parameter that you must specify, 
        error is the error calculated by the back-propagation procedure for the neuron and 
        input is the input value that caused the error.
        '''
        for i in range(len(network)):
            inputs = row[:-1]
            if i != 0:
                inputs = [neuron['output'] for neuron in network[i - 1]]
            for neuron in network[i]:
                for j in range(len(inputs)):
                    neuron['weights'][j] += l_rate * neuron['delta'] * inputs[j]
                neuron['weights'][-1] += l_rate * neuron['delta']
    
    def train_network(self, network, activation_function, train, l_rate, n_epoch, n_outputs):
        '''
        Train a network for a fixed number of epochs.
        The network is updated using stochastic gradient descent.
        '''
        for epoch in range(n_epoch + 1):
            sum_error = 0
            for row in train:
                # Calculate Loss
                outputs = self.forward_propagate(network, activation_function, row)
                expected = [0 for i in range(n_outputs)]
                expected[row[-1]] = 1  # Bias
                sum_error += sum([(expected[i] - outputs[i]) ** 2 for i in range(len(expected))])
                self.backward_propagate_error(network, activation_function, expected)
                self.update_weights(network, row, l_rate)
            if (epoch % 100 == 0):    
                print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, l_rate, sum_error))
    
    def predict(self, network, activationFunction, row):
        '''
        Make a prediction with a network.
        We can use the output values themselves directly as the probability of a pattern belonging to each output class.
        It may be more useful to turn this output back into a crisp class prediction. 
        We can do this by selecting the class value with the larger probability. 
        This is also called the arg max function.
        '''
        outputs = self.forward_propagate(network, activationFunction, row)
        return outputs.index(max(outputs))
    
    def back_propagation(self, train, test, l_rate, n_epoch, n_hidden, activationFunction):
        '''
        Backpropagation Algorithm With Stochastic Gradient Descent
        '''
        n_inputs = len(train[0]) - 1
        n_outputs = len(set([row[-1] for row in train]))
        network = self.initialize_network(n_inputs, n_hidden, n_outputs)
        self.train_network(network, activationFunction, train, l_rate, n_epoch, n_outputs)
        predictions = list()
        for row in test:
            prediction = self.predict(network, activationFunction, row)
            predictions.append(prediction)
        return(predictions)

if __name__ == '__main__':    
    
    seed(1)
    mlp = MultilayerNnClassifier()
    activationFunction = Tanh()
    network = mlp.initialize_network(2, 1, 2)
    for layer in network:
        print(layer)
        
    # Test forward_propagate
    print("Test Forward")
    row = [1, 0, None]
    output = mlp.forward_propagate(network, activationFunction, row)
    print(output)
    
    # Test backward_propagate_error
    print("Test backpropagation of error")
    network = [[{'output': 0.7105668883115941, 'weights': [0.13436424411240122, 0.8474337369372327, 0.763774618976614]}],
            [{'output': 0.6213859615555266, 'weights': [0.2550690257394217, 0.49543508709194095]}, {'output': 0.6573693455986976, 'weights': [0.4494910647887381, 0.651592972722763]}]]
    expected = [0, 1]
    mlp.backward_propagate_error(network, activationFunction, expected)
    for layer in network:
        print(layer)
        
    # Test training backprop algorithm
    print("Test training backprop algorithm")
    seed(1)
    dataset = [[2.7810836, 2.550537003, 0],
        [1.465489372, 2.362125076, 0],
        [3.396561688, 4.400293529, 0],
        [1.38807019, 1.850220317, 0],
        [3.06407232, 3.005305973, 0],
        [7.627531214, 2.759262235, 1],
        [5.332441248, 2.088626775, 1],
        [6.922596716, 1.77106367, 1],
        [8.675418651, -0.242068655, 1],
        [7.673756466, 3.508563011, 1]]
    n_inputs = len(dataset[0]) - 1
    n_outputs = len(set([row[-1] for row in dataset]))
    network = mlp.initialize_network(n_inputs, 2, n_outputs)
    mlp.train_network(network, activationFunction, dataset, 0.5, 20, n_outputs)    
    for layer in network:
        print(layer)
    for row in dataset:
        prediction = mlp.predict(network, activationFunction, row)
        print('Expected=%d, Got=%d' % (row[-1], prediction))     
