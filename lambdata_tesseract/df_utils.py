"""
Utility function for working with DataFrames.

"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

TEST_DF = pd.DataFrame({'A': [1, 2, 3, 4, np.nan, 6, 7], 
                        'B': [8, np.nan, 10, np.nan, 12, np.nan, 14]})


class HelperFunctions:
  """
  Useful Data Science Helper Functions.
  """
  
  def __init__(self):
    pass
    
  def nulls(self, df):
    """
    See how many null values are in each column of a dataframe.
    """
    self.df = df
    nulls_df = pd.DataFrame(df.isnull().sum(), columns=['Null Value Count'])
    return nulls_df
  
  def train_validation_test_split(self, X, y, train_size = 0.8, val_size = 0.1, 
                                  test_size = 0.1, random_state = None, 
                                  shuffle = True):
    """
    Splitting data into train, validation and test datasets. Must enter X and y
    variables. Other parameter defaults: train_size = 0.8, val_size = 0.1, 
    test_size = 0.1, random_state = None, shuffle = True 
    """
    self.X = X
    self.y = y
    self.train_size = train_size
    self.val_size = val_size
    self.test_size = test_size
    self.random_state = random_state
    self.shuffle = shuffle
    
    assert train_size + val_size + test_size == 1
    
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, shuffle=shuffle)
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_size/(train_size+val_size), 
        random_state=random_state, shuffle=shuffle)
    
    return X_train, X_val, X_test, y_train, y_val, y_test
