"""
Utility functions
"""
import pandas as pd
import numpy as np

def split_dataset(df, ratio, random_state=18, indices=False):
    """
    Splits a pandas dataframe into train and test sets
    Params:
    df:     The data
    ratio:  Fraction representing how much of the original dataset should be left
            for training (e.g, 0.7 for a 70/30 split)
    """
    train = df.sample(frac=ratio, random_state=random_state)
    train_idx = train.index
    test = df.drop(train_idx)
    test_idx = test.index
    test = test.reset_index(drop=True)
    train = train.reset_index(drop=True)

    if indices:

        train_col= pd.DataFrame({'index':train_idx.values, 'train_test':np.repeat("train",len(train_idx.values))})
        test_col= pd.DataFrame({'index':test_idx.values, 'train_test':np.repeat("test",len(test_idx.values))})
        train_test = pd.concat([train_col,test_col])
        train_test = train_test.sort_values(by=['index'])
        train_test = train_test.drop(['index'],axis=1)
        train_test = train_test.reset_index(drop=True)

        return train, test, train_test

    return train, test
