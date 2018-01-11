'''
Created on 09 gen 2018

@author: Umberto Griffo
'''

from ml.activation import ActivationFunction

class ReLU(ActivationFunction.ActivationFunction):

    def transfer(self, activation):
        '''
        Rectified Linear Unit activation function.
        '''
        if(activation < 0):
            return 0
        elif(activation >= 0):
            return activation
    
    def transfer_derivative(self, output):
        '''
        We are using the Rectified Linear Unit transfer function, the derivative of which can be calculated as follows:
        derivative = 0 for x<0 ; 1 for x>=0
        '''
        if(output < 0):
            return 0
        elif(output >= 0):
            return 1
