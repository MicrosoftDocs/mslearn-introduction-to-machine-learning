from typing import List
import sklearn.model_selection as model_selection
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import pandas as pd


def train_logistic_regression(all_data:pd.DataFrame, features:List[str]):
    '''
    Trains a model to predict survival on the Titanic
    '''

    # Split our dataset into training and test datasets
    train, test = model_selection.train_test_split(all_data, test_size=0.30, random_state=101)

    # X is our feature matrix
    X = train[features]

    # y is the label vector 
    y = train.Survived

    # train the model
    model = LogisticRegression(random_state=0, max_iter=2000).fit(X, y)

    # calculate loss
    probabilities = model.predict_proba(test[features])
    loss = metrics.log_loss(test.Survived, probabilities)

    return loss