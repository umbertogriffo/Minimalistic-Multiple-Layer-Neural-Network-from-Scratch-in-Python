'''
Created on 09 gen 2018

@author: Umberto

'''
from math import sqrt

class RegressionEvaluator:
        
    def mse_metric(self, actual, predicted):
        '''
        Calculate mse
        '''
        s = 0
        for i in range(len(actual)):
            a = actual[i]
            p = predicted[i]
            s += (a - p) * (a - p)
        return s / float(len(actual))
    
    def rmse_metric(self, actual, predicted):
        '''
        Calculate rmse
        '''
        s = 0
        for i in range(len(actual)):
            a = actual[i]
            p = predicted[i]
            s += (a - p) * (a - p)
        return sqrt(s / float(len(actual)))
    
    def r2_metric(self, actual, predicted):
        '''
        Calculate r2
        '''
        s = 0
        sumMean = 0
        mean = sum(actual) / float(len(actual))
        for i in range(len(actual)):
            a = actual[i]
            p = predicted[i]
            s += (a - p) * (a - p)
            sumMean += (a - mean) * (a - mean)
        return 1 - (s / sumMean)
    
    def evaluate_algorithm(self, dataset,splitting, algorithm, n_folds, minmax, *args):
        '''
        Evaluate an algorithm using a cross validation split
        '''
        folds = splitting.cross_validation_split(dataset, n_folds)
        scores = list()
        for fold in folds:
            train_set = list(folds)
            train_set.remove(fold)
            train_set = sum(train_set, [])
            test_set = list()
            for row in fold:
                row_copy = list(row)
                test_set.append(row_copy)
                row_copy[-1] = None
                
            predicted = algorithm(train_set, test_set, *args)
            # Rescale the predictions
            self.denormalize_target(predicted, minmax)
            
            actual = [row[-1] for row in fold]
            # Rescale the actuals
            self.denormalize_target(actual, minmax)
            
            mse = self.mse_metric(actual, predicted)
            rmse = self.rmse_metric(actual, predicted)
            r2 = self.r2_metric(actual, predicted)
            
            print('>mse=%.3f, rmse=%.3f, r2=%.3f' % (mse, rmse, r2))
            scores.append((mse, rmse, r2))
        return scores   

    def denormalize_target(self, target, target_minmax):
        '''
        Rescale predicted Value to the original range
        '''    
        for i in range(len(target)):
            target[i] = (target[i] * (target_minmax[1] - target_minmax[0])) + target_minmax[0]             
