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
        
    seed(1)
    
    n_folds = 5
    l_rate = 0.3
    n_epoch = 1000
    n_hidden = [10]
    
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
    print_classification_scores(scores) 

def classificationWineRed():
    '''
    Test Classification on WineRed dataset
    '''
        
    seed(1)
    
    n_folds = 5
    l_rate = 0.3
    n_epoch = 1000
    n_hidden = [10]
    
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
    print_classification_scores(scores)  
    
def classificationPokemon():
    '''
    Test Classification on Pokemon dataset
    id_combat    pk1_ID    pk1_Name    pk1_Type1    pk1_Type2    pk1_HP    pk1_Attack    pk1_Defense    pk1_SpAtk
    pk1_SpDef    pk1_Speed    pk1_Generation    pk1_Legendary    pk1_Grass    pk1_Fire    pk1_Water    pk1_Bug    
    pk1_Normal    pk1_Poison    pk1_Electric    pk1_Ground    pk1_Fairy    pk1_Fighting    pk1_Psychic    pk1_Rock    
    pk1_Ghost    pk1_Ice    pk1_Dragon    pk1_Dark    pk1_Steel    pk1_Flying    ID    pk2_Name    pk2_Type1    pk2_Type2    
    pk2_HP    pk2_Attack    pk2_Defense    pk2_SpAtk    pk2_SpDef    pk2_Speed    pk2_Generation    pk2_Legendary    
    pk2_Grass    pk2_Fire    pk2_Water    pk2_Bug    pk2_Normal    pk2_Poison    pk2_Electric    pk2_Ground    pk2_Fairy    
    pk2_Fighting    pk2_Psychic    pk2_Rock    pk2_Ghost    pk2_Ice    pk2_Dragon    pk2_Dark    pk2_Steel    pk2_Flying    winner                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        
    '''
        
    seed(1)
    
    n_folds = 5
    l_rate = 0.1
    n_epoch = 500
    n_hidden = [5]
    
    mlp = MultilayerNnClassifier();
    activationFunction = Sigmoid()
    dp = DataPreparation();
    evaluator = ClassificationEvaluator();
    splitting = Splitting();
   
    # load and prepare data
    filename = '../Datasets/pkmn.csv'
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
    print_classification_scores(scores) 
    
def regressionWineRed():
    '''
    Test Regression on WineRed dataset
    '''
        
    seed(1)
    
    n_folds = 5
    l_rate = 0.3
    n_epoch = 1000
    n_hidden = [20,10]
    
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
    print_regression_scores(scores)

def regressionWineWhite():
    '''
    Test Classification on WineWhite dataset
    '''
        
    seed(1)
    
    n_folds = 5
    l_rate = 0.3
    n_epoch = 1000
    n_hidden = [10,5]
    
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
    print_regression_scores(scores)

def print_regression_scores(scores):
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

def print_classification_scores(scores):
    print('Scores: %s' % scores)
    print('Mean Accuracy: %.3f%%' % (sum(scores) / float(len(scores))))   
        
if __name__ == '__main__':
    
    options = {
           0 : classificationSeed,
           1 : classificationWineRed,
           2 : classificationPokemon,
           3 : regressionWineRed,
           4 : regressionWineWhite
           }
    
    var = input("Please enter one of following numbers: \n 0 - Classification on Seed Dataset\n 1 - Classification on Wine Red Dataset\n 2 - Classification on Pokemon Dataset\n 3 - Regression on Wine White Dataset\n 4 - Regression on Wine Red Dataset \n")
    print("You entered " + str(var))
    if int(var) >= len(options) or int(var) < 0:
        raise Exception('You have entered an invalid number: ' +str(len(options)))
    options[int(var)]()    
