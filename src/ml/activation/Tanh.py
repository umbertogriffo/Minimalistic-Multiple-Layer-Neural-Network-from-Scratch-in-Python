'''
Created on 09 gen 2018

@author: Umberto Griffo
'''

from ml.activation import ActivationFunction
from math import exp

class Tanh(ActivationFunction.ActivationFunction):

    def transfer(self, activation):
        '''
        Tanh activation function.
        '''
        return (exp(activation) - exp(-activation))/(exp(activation) + exp(-activation))
    
    def transfer_derivative(self, output):
        '''
        We are using the tanh transfer function, the derivative of which can be calculated as follows:
        derivative = 1 - (output * output)
        '''
        return 1 - (output * output)