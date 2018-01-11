'''
Created on 09 gen 2018

@author: Umberto Griffo

'''

from random import randrange

class Splitting:
        
    def cross_validation_split(self, dataset, n_folds):
        '''
        Split a dataset into k folds
        '''        
        dataset_split = list()
        dataset_copy = list(dataset)
        fold_size = int(len(dataset) / n_folds)
        for i in range(n_folds):
            fold = list()
            while len(fold) < fold_size:
                index = randrange(len(dataset_copy))
                fold.append(dataset_copy.pop(index))
            dataset_split.append(fold)
        return dataset_split