'''
Created on 09 gen 2018

@author: Umberto
'''

from random import seed

from ml.MultilayerNnRegressor import MultilayerNnRegressor
from ml.MultilayerNnClassifier import MultilayerNnClassifier
from ml.activation.Sigmoid import Sigmoid
from DataPreparation import DataPreparation
from evaluation.ClassificationEvaluator import ClassificationEvaluator
from evaluation.RegressionEvaluator import RegressionEvaluator
from evaluation.Splitting import Splitting

def classificationSeed():
    '''
    Test Classification on Seeds dataset
    '''
    mlp = MultilayerNnClassifier();
    activationFunction = Sigmoid()
    dp = DataPreparation();
    evaluator = ClassificationEvaluator();
    splitting = Splitting();
   
    # load and prepare data
    filename = '../Datasets/seeds_dataset.csv'
    dataset = dp.load_csv(filename)
    for i in range(len(dataset[0]) - 1):
        dp.str_column_to_float(dataset, i)
    # convert class column to integers
    dp.str_column_to_int(dataset, len(dataset[0]) - 1)
    # normalize input variables
    minmax = dp.dataset_minmax(dataset)
    dp.normalize_dataset_classification(dataset, minmax)    
    # evaluate algorithm
    scores = evaluator.evaluate_algorithm(dataset, splitting, mlp.back_propagation, n_folds, l_rate, n_epoch, n_hidden, activationFunction)  
    print('Scores: %s' % scores)
    print('Mean Accuracy: %.3f%%' % (sum(scores) / float(len(scores))))   

def classificationWineRed():
    '''
    Test Classification on WineRed dataset
    '''
    mlp = MultilayerNnClassifier();
    activationFunction = Sigmoid()
    dp = DataPreparation();
    evaluator = ClassificationEvaluator();
    splitting = Splitting();
        
    # Test Backprop on Seeds dataset
    # load and prepare data
    filename = '../Datasets/winequality-red.csv'
    dataset = dp.load_csv(filename)
    for i in range(len(dataset[0]) - 1):
        dp.str_column_to_float(dataset, i)
    # convert class column to integers
    dp.str_column_to_int(dataset, len(dataset[0]) - 1)
    # normalize input variables
    minmax = dp.dataset_minmax(dataset)
    dp.normalize_dataset_classification(dataset, minmax)    
    # evaluate algorithm
    scores = evaluator.evaluate_algorithm(dataset, splitting, mlp.back_propagation, n_folds, l_rate, n_epoch, n_hidden, activationFunction)  
    print('Scores: %s' % scores)
    print('Mean Accuracy: %.3f%%' % (sum(scores) / float(len(scores))))   

def regressionWineRed():
    '''
    Test Regression on WineRed dataset
    '''
    mlp = MultilayerNnRegressor();
    activationFunction = Sigmoid()
    dp = DataPreparation();
    evaluator = RegressionEvaluator();
    splitting = Splitting();
   
    # load and prepare data
    filename = '../Datasets/winequality-red.csv'
    dataset = dp.load_csv(filename)
    for i in range(len(dataset[0])):
        dp.str_column_to_float(dataset, i)
    # normalize input variables including the target
    minmax = dp.dataset_minmax(dataset)
    target_minmax = minmax[-1]
    dp.normalize_dataset_regression(dataset, minmax)    
    # evaluate algorithm
    scores = evaluator.evaluate_algorithm(dataset, splitting, mlp.back_propagation, n_folds, target_minmax, l_rate, n_epoch, n_hidden, activationFunction, target_minmax) 
    print('Scores: %s' % scores)
    sum_mse = 0
    sum_rmse = 0
    sum_r2 = 0
    for score in scores:
        sum_mse += score[0]
        sum_rmse += score[1]
        sum_r2 += score[2]         
    print('Mean MSE: %.3f' % (sum_mse / float(len(scores))))
    print('Mean RMSE: %.3f' % (sum_rmse / float(len(scores))))
    print('Mean R^2: %.3f' % (sum_r2 / float(len(scores)))) 

def regressionWineWhite():
    '''
    Test Classification on WineWhite dataset
    '''
    mlp = MultilayerNnRegressor();
    activationFunction = Sigmoid()
    dp = DataPreparation();
    evaluator = RegressionEvaluator();
    splitting = Splitting();

    # load and prepare data
    filename = '../Datasets/winequality-white.csv'
    dataset = dp.load_csv(filename)
    for i in range(len(dataset[0])):
        dp.str_column_to_float(dataset, i)
    # normalize input variables including the target
    minmax = dp.dataset_minmax(dataset)
    target_minmax = minmax[-1]
    dp.normalize_dataset_regression(dataset, minmax)    
    # evaluate algorithm
    scores = evaluator.evaluate_algorithm(dataset, splitting , mlp.back_propagation, n_folds, target_minmax, l_rate, n_epoch, n_hidden, activationFunction, target_minmax) 
    print('Scores: %s' % scores)
    sum_mse = 0
    sum_rmse = 0
    sum_r2 = 0
    for score in scores:
        sum_mse += score[0]
        sum_rmse += score[1]
        sum_r2 += score[2]         
    print('Mean MSE: %.3f' % (sum_mse / float(len(scores))))
    print('Mean RMSE: %.3f' % (sum_rmse / float(len(scores))))
    print('Mean R^2: %.3f' % (sum_r2 / float(len(scores)))) 

if __name__ == '__main__':
    
    seed(1)
    
    n_folds = 5
    l_rate = 0.3
    n_epoch = 1000
    n_hidden = 10
    
    # map the inputs to the function blocks
    options = {0 : classificationSeed,
           1 : classificationWineRed,
           2 : regressionWineRed,
           3 : regressionWineWhite
           }
    
    var = input("Please enter one of following numbers: \n 0 - Classification on Seed Dataset\n 1 - Classification on Wine Red Dataset\n 2 - Regression on Wine Red Dataset\n 3 - Regression on Wine White Dataset\n")
    print("You entered " + str(var))
    if int(var) >= len(options):
        raise Exception('You have to enter a number < ' +str(len(options)))
    options[int(var)]()
    
