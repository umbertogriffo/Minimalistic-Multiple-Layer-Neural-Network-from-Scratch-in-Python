# Minimalistic Multiple Layer Neural Network from Scratch in Python
* Author: Umberto Griffo

Inspired by [1] and [2] I implemented a *Minimalistic Multiple Layer Neural Network* from Scratch in Python.
You can use It to better understand the core concepts of Neural Network.

## Software Environment
* Python 3.0 - 3.5

## Features
- Backpropagation Algorithm With *Stochastic Gradient Descent*. During training we are using single training examples for one forward/backward pass.
- *Classification* (MultilayerNnClassifier.py).
- *Regression* (MultilayerNnRegressor.py).
- *Activation Function*: Linear, ReLU, Sigmoid, Tanh.
- *Classification Evaluator*: Accuracy.
- *Regression Evaluator*: Mean Squared Error (MSE), Root Mean Squared Error (RMSE), Coefficient of Determination (R^2).

## Possible Extensions:
- *Early stopping*.
- Experiment with different *weight initialization techniques* (such as small random numbers).
- *More Layers*. Add support for more hidden layers.
- *Batch Gradient Descent*. Change the training procedure from online to batch gradient descent 
  and update the weights only at the end of each epoch.
- *Mini-Batch Gradient Descent*. More info [here](http://cs231n.github.io/optimization-1/#gd).
- *Momentum*. More info [here](http://cs231n.github.io/neural-networks-3/#update).
- *Annealing the learning rate*. More info [here](http://cs231n.github.io/neural-networks-3/#anneal).
- *Dropout Regularization*, *Batch Normalization*. More info [here](http://cs231n.github.io/neural-networks-2/).
- *Model Ensembles*. More info [here](http://cs231n.github.io/neural-networks-3/).

## References:
- [1] How to Implement Backpropagation Algorithm from scratch in Python [here](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/).
- [2] Implementing Multiple Layer Neural Network from Scratch [here](https://github.com/pangolulu/neural-network-from-scratch).
- [3] Andrew Ng Lecture on *Gradient Descent* [here](http://cs229.stanford.edu/notes/cs229-notes1.pdf).
- [4] Andrew Ng Lecture on *Backpropagation Algorithm* [here](http://cs229.stanford.edu/notes/cs229-notes-backprop.pdf).
- [5] (P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis. 
Modeling wine preferences by data mining from physicochemical properties. In Decision Support Systems, Elsevier, 47(4):547-553, 2009.) [here](https://archive.ics.uci.edu/ml/datasets/wine+quality)
- [6] seeds Data Set [here](http://archive.ics.uci.edu/ml/datasets/seeds)


