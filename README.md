# Minimalistic Multiple Layer Neural Network from Scratch in Python
* Author: Umberto Griffo

Inspired by <a href="https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/">[1]</a> and <a href="https://github.com/pangolulu/neural-network-from-scratch">[2]</a> I implemented a **Minimalistic Multiple Layer Neural Network** from Scratch in Python.
You can use It to better understand the core concepts of Neural Networks.

## Software Environment
* Python 3.0 - 3.5

## Features
- Backpropagation Algorithm With **Stochastic Gradient Descent**. During training we are using single training examples for one forward/backward pass.
- Supporting multiple hidden layers.
- **Classification** (MultilayerNnClassifier.py).
- **Regression** (MultilayerNnRegressor.py).
- **Activation Function**: Linear, ReLU, Sigmoid, Tanh.
- **Classification Evaluator**: Accuracy.
- **Regression Evaluator**: Mean Squared Error (MSE), Root Mean Squared Error (RMSE), Coefficient of Determination (R^2).

## Demo
If you run Test.py you can see the following textual menu:
```
Please enter one of following numbers: 
 0 - Classification on Seed Dataset
 1 - Classification on Wine Red Dataset
 2 - Classification on Pokemon Dataset
 3 - Regression on Wine White Dataset
 4 - Regression on Wine Red Dataset 
```
If you choose 2 will be performed a classification task on Pokemon Dataset:
```
2
You entered 2
>epoch=0, lrate=0.100, error=0.396
>epoch=100, lrate=0.100, error=0.087
>epoch=200, lrate=0.100, error=0.083
>epoch=300, lrate=0.100, error=0.081
>epoch=400, lrate=0.100, error=0.081
>epoch=500, lrate=0.100, error=0.080
>accuracy=95.450
>epoch=0, lrate=0.100, error=0.353
>epoch=100, lrate=0.100, error=0.092
>epoch=200, lrate=0.100, error=0.085
>epoch=300, lrate=0.100, error=0.083
>epoch=400, lrate=0.100, error=0.082
>epoch=500, lrate=0.100, error=0.081
>accuracy=95.400
>epoch=0, lrate=0.100, error=0.415
>epoch=100, lrate=0.100, error=0.087
>epoch=200, lrate=0.100, error=0.083
>epoch=300, lrate=0.100, error=0.082
>epoch=400, lrate=0.100, error=0.081
>epoch=500, lrate=0.100, error=0.080
>accuracy=95.520
>epoch=0, lrate=0.100, error=0.401
>epoch=100, lrate=0.100, error=0.089
>epoch=200, lrate=0.100, error=0.084
>epoch=300, lrate=0.100, error=0.083
>epoch=400, lrate=0.100, error=0.082
>epoch=500, lrate=0.100, error=0.081
>accuracy=95.280
>epoch=0, lrate=0.100, error=0.395
>epoch=100, lrate=0.100, error=0.093
>epoch=200, lrate=0.100, error=0.087
>epoch=300, lrate=0.100, error=0.085
>epoch=400, lrate=0.100, error=0.084
>epoch=500, lrate=0.100, error=0.083
>accuracy=94.900
Scores: [95.45, 95.39999999999999, 95.52000000000001, 95.28, 94.89999999999999]
Mean Accuracy: 95.310%
```

## Possible Extensions:
- **Early stopping**.
- Experiment with different **weight initialization techniques** (such as small random numbers).
- **Batch Gradient Descent**. Change the training procedure from online to batch gradient descent 
  and update the weights only at the end of each epoch.
- **Mini-Batch Gradient Descent**. More info [here](http://cs231n.github.io/optimization-1/#gd).
- **Momentum**. More info [here](http://cs231n.github.io/neural-networks-3/#update).
- **Annealing the learning rate**. More info [here](http://cs231n.github.io/neural-networks-3/#anneal).
- **Dropout Regularization**, **Batch Normalization**. More info [here](http://cs231n.github.io/neural-networks-2/).
- **Model Ensembles**. More info [here](http://cs231n.github.io/neural-networks-3/).

## References:
- [1] How to Implement Backpropagation Algorithm from scratch in Python [here](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/).
- [2] Implementing Multiple Layer Neural Network from Scratch [here](https://github.com/pangolulu/neural-network-from-scratch).
- [3] Andrew Ng Lecture on *Gradient Descent* [here](http://cs229.stanford.edu/notes/cs229-notes1.pdf).
- [4] Andrew Ng Lecture on *Backpropagation Algorithm* [here](http://cs229.stanford.edu/notes/cs229-notes-backprop.pdf).
- [5] (P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis. 
Modeling wine preferences by data mining from physicochemical properties. In Decision Support Systems, Elsevier, 47(4):547-553, 2009.) [here](https://archive.ics.uci.edu/ml/datasets/wine+quality)
- [6] seeds Data Set [here](http://archive.ics.uci.edu/ml/datasets/seeds)


