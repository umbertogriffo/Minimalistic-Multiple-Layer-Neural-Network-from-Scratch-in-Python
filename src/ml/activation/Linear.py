'''
Created on 09 gen 2018

@author: Umberto Griffo
'''

from ml.activation import ActivationFunction

class Linear(ActivationFunction.ActivationFunction):

    def transfer(self, activation):
        '''
        Linear activation function.
        '''
        return activation 
    
    def transfer_derivative(self, output):
        '''
        We are using the linear transfer function, the derivative of which can be calculated as follows:
        derivative = 1.0
        '''
        return 1.0