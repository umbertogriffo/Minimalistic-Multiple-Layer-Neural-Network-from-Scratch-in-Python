'''
Created on 09 gen 2018

@author: Umberto Griffo
'''

from ml.activation import ActivationFunction
from math import exp

class Sigmoid(ActivationFunction.ActivationFunction):

    def transfer(self, activation):
        '''
        Sigmoid activation function.
        '''
        return 1.0 / (1.0 + exp(-activation))
    
    def transfer_derivative(self, output):
        '''
        We are using the sigmoid transfer function, the derivative of which can be calculated as follows:
        derivative = output * (1.0 - output)
        '''
        return output * (1.0 - output)