import sklearn.model_selection as model_selection
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import pandas as pd

test_passenger_ids = None
features = None
X_test = None
y_test = None

def prepare_test_set(data_features):
    '''
    Extracts a test dataset for later use
    '''
    global test_passenger_ids, features, X_test, y_test

    # Make a dataset we will test both models on. This should be 
    # data that, before cleaning, did not lack any of the features
    # we are interested in. It is important that this test dataset
    # is not cleaned, as our cleaning process falsifies some values
    # which could distort performance to look better than it
    # actually is

    # Load the full dataset
    dataset = pd.read_csv('Data/titanic.csv', index_col=False, sep=",", header=0)


    # One-hot encode categorical variables
    dataset = pd.get_dummies(dataset, columns=["Pclass", "Sex", "Cabin", "Embarked"], drop_first=False)

    # Add the Unknown cabin column
    # This is blank for our test data because our
    # test data does not have anyone missing info
    ca = pd.options.mode.chained_assignment
    pd.options.mode.chained_assignment = None # suppresses warning
    dataset['Cabin_Unknown'] = 0
    pd.options.mode.chained_assignment = ca # roll back

    features = [value for value in data_features if value in dataset.columns]
    keep_features = list(features)
    keep_features.append("PassengerId")
    keep_features.append("Survived")
    # Find passengers where all features are not null
    eligible_rows = (~dataset[keep_features].isnull()).sum(axis=1) == len(keep_features)
    passengers_with_full_information = dataset[eligible_rows].PassengerId

    # Randomly select some passenger Ids
    _, test_passenger_ids = model_selection.train_test_split(passengers_with_full_information, test_size=0.30, random_state=101)

    # Extract our feature columns for these passengers,
    # giving us our test dataset
    test_rows = dataset[dataset.PassengerId.isin(test_passenger_ids)]
    # print(features)
    # features.remove("PassengerId")
    # features.remove("Survived")
    X_test = test_rows[features]
    y_test = test_rows.Survived


def train_logistic_regression(data):
    '''
    Trains a model to predict survival on the Titanic
    This is tested against a test dataset that had no data missing 
    '''
    global test_passenger_ids, features, X_test, y_test

    # Extract training data from the dataset
    # This is all passengers who are not in the test set
    data = data[~data.PassengerId.isin(test_passenger_ids)]

    # X is our feature matrix
    X = data[features]

    # y is the label vector 
    y = data.Survived

    # train the model
    model = LogisticRegression(random_state=0, max_iter=2000).fit(X, y)

    # score is the mean accuracy on the given test data and labels
    score = model.score(X, y)

    # calculate loss
    probabilities = model.predict_proba(X_test)
    loss = metrics.log_loss(y_test, probabilities)

    return score, loss