'''
Created on 09 gen 2018

@author: Umberto

'''
class ClassificationEvaluator:
        
    def accuracy_metric(self, actual, predicted):
        '''
        Calculate accuracy percentage
        '''
        correct = 0
        for i in range(len(actual)):
            if actual[i] == predicted[i]:
                correct += 1
        return correct / float(len(actual)) * 100.0
    
    def evaluate_algorithm(self, dataset, splitting, algorithm, n_folds, *args):
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
            actual = [row[-1] for row in fold]
            accuracy = self.accuracy_metric(actual, predicted)
            scores.append(accuracy)
        return scores   
