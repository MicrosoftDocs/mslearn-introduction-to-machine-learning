"""
Utility functions
"""

def split_dataset(df, ratio):
    """
    Splits a pandas dataframe into train and test sets

    Params:

    df:     The data
    ratio:  Fraction representing how much of the original dataset should be left
            for training (e.g, 0.7 for a 70/30 split)
    """
    train = df.sample(frac=ratio, random_state=18)
    test = df.drop(train.index).reset_index(drop=True)
    train = train.reset_index(drop=True)
    return train, test
