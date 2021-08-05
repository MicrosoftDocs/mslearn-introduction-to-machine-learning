'''
This material is explain in the notebooks. We list it here so it can be reused across exercises
'''
import numpy
import pandas
import graphing # Custom graphing code

def calculate_tpr_fpr(prediction, actual):
    '''
    Calculates true positive rate and false positive rate

    prediction: the labels predicted by the model
    actual:     the correct labels we hope the model predicts
    '''

    # To calculate the true positive rate and true negative rate we need to know
    # TP - how many true positives (where the model predicts hiker, and it is a hiker)
    # TN - how many true negatives (where the model predicts tree, and it is a tree)
    # FP - how many false positives (where the model predicts hiker, but it was a tree)
    # FN - how many false negatives (where the model predicts tree, but it was a hiker)

    # First, make a note of which predictions were 'true' and which were 'false'
    prediction_true = numpy.equal(prediction, 1)
    prediction_false= numpy.equal(prediction, 0)

    # Now, make a note of which correct results were 'true' and which were 'false'
    actual_true = numpy.equal(actual, 1)
    actual_false = numpy.equal(actual, 0)

    # Calculate TP, TN, FP, and FN
    # The combination of sum and '&' counts the overlap
    # For example, TP calculates how many 'true' predictions 
    # overlapped with 'true' labels (correct answers)
    TP = numpy.sum(prediction_true  & actual_true)
    TN = numpy.sum(prediction_false & actual_false)
    FP = numpy.sum(prediction_true  & actual_false)
    FN = numpy.sum(prediction_false & actual_true)

    # Calculate the true positive rate
    # This is the proportion of 'hiker' labels that are identified as hikers
    tpr = TP / (TP + FN)

    # Calculate the false positive rate 
    # This is the proportion of 'tree' labels that are identified as hikers
    fpr = FP / (FP + TN)

    # Return both rates
    return tpr, fpr


def assess_model(model_predict, test, feature_name, threshold):
    '''
    Calculates the true positive rate and false positive rate of the model
    at a particular decision threshold

    model_predict: the model's predict function
    test: the test dataset
    feature_name: the feature the model is expecting
    threshold: the decision threshold to use 
    '''

    # Make model predictions for every sample in the test set
    # What we get back is a probability that the sample is a hiker
    # For example, if we had two samples in the test set, we might
    # get 0.45 and 0.65, meaning the model says there is a 45% chance
    # the first sample is a hiker, and 65% chance the second is a 
    # hiker
    probability_of_hiker = model_predict(test[feature_name])
    
    # See which predictions at this threshold would say hiker
    predicted_is_hiker = probability_of_hiker > threshold

    # calculate the true and false positives rates using our
    # handy method
    return calculate_tpr_fpr(predicted_is_hiker, test.is_hiker)


def create_roc_curve(model_predict, test, feature="motion"):
    '''
    This function creates a ROC curve for a given model by testing it
    on the test set for a range of decision thresholds. An ROC curve has
    the True Positive rate on the x-axis and False Positive rate on the 
    y-axis

    model_predict: The model's predict function, returning probabilities
    test: the test set
    feature: The feature to provide the model's predict function

    Returns the plotly figure and a dataframe of the results
    '''

    # Calculate what the true positive and false positive rate would be if
    # we had used different thresholds. 

    #  Make a list of thresholds to try
    # NB We need some specific values for our exercises but this is not typical
    thresholds = numpy.sort(numpy.hstack([[-1E-6], [1.0001], [0.5], [0.3], numpy.linspace(0,1,100)]))

    false_positive_rates = []
    true_positive_rates = []

    # Loop through all thresholds
    for threshold in thresholds:
        # calculate the true and false positives rates using our
        # handy method
        tpr, fpr = assess_model(model_predict, test, feature, threshold)

        # save the results
        true_positive_rates.append(tpr)
        false_positive_rates.append(fpr)


    # Graph the result
    # You don't need to understand this code, but essentially we are plotting
    # TPR versus FPR as a line plot
    # -- Prepare a dataframe, required by our graphing code
    df_for_graphing = pandas.DataFrame(dict(threshold=thresholds, fpr=false_positive_rates, tpr=true_positive_rates))
    # -- Generate the plot
    fig = graphing.scatter_2D(df_for_graphing, label_x="fpr", label_y="tpr", x_range=[-0.05,1.05])
    fig.update_traces(mode='lines') # Comment our this line if you would like to see points rather than lines
    fig.update_yaxes(range=[-0.05, 1.05])

    # Return the graph
    return fig, df_for_graphing
