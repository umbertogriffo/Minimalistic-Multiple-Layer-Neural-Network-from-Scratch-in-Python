'''
Created on 09 gen 2018

@author: Umberto Griffo
'''

from csv import reader

class DataPreparation:
     
    def load_csv(self, filename):
        '''
        Load a CSV file
        '''
        dataset = list()
        with open(filename, 'r') as file:
            csv_reader = reader(file)
            for row in csv_reader:
                if not row:
                    continue
                dataset.append(row)
        return dataset
    
    def str_column_to_float(self, dataset, column):
        '''
        Convert string column to float
        '''
        for row in dataset:
            row[column] = float(row[column].strip())
     
    def str_column_to_int(self, dataset, column):
        '''
        Convert string column to integer
        '''
        class_values = [row[column] for row in dataset]
        unique = set(class_values)
        lookup = dict()
        for i, value in enumerate(unique):
            lookup[value] = i
        for row in dataset:
            row[column] = lookup[row[column]]
        return lookup
    
    def dataset_minmax(self, dataset):
        '''
        Find the min and max values for each column
        '''    
        stats = [[min(column), max(column)] for column in zip(*dataset)]
        self.stats = stats
        return stats

    def normalize_dataset_classification(self, dataset, minmax):
        '''
        Rescale dataset columns to the range 0-1
        '''    
        for row in dataset:
            for i in range(len(row) - 1):
                row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])
    
    def denormalize_dataset_classification(self, dataset, minmax):
        '''
        Rescale dataset columns to the original range
        '''    
        for row in dataset:
            for i in range(len(row) - 1):
                row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])            
    
    def normalize_dataset_regression(self, dataset, minmax):
        '''
        Rescale dataset columns to the range 0-1
        '''    
        for row in dataset:
            for i in range(len(row)):
                row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])   
    
    def denormalize_dataset_regression(self, dataset, minmax):
        '''
        Rescale dataset columns to the original range
        '''    
        for row in dataset:
            for i in range(len(row)):
                row[i] = (row[i] * (minmax[i][1] - minmax[i][0])) + minmax[i][0]                           