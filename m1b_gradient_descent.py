import numpy

class MyModel:

    def __init__(self):
        '''
        Creates a new MyModel
        '''
        # Straight lines described by two parameters:
        # The slop is the angle of the line
        self.slope = 0
        # The intercept moves the line up or down
        self.intercept = 0
        # The history of cost per iteration
        self.cost_history = []

    def predict(self, x):
        '''
        Estimates the target variable from the value of x
        '''
        return x * self.slope + self.intercept

    def get_summary(self):
        '''
        Returns a string that summarises the model
        '''
        return f"y = {self.slope} * x + {self.intercept}"


def cost_function(actual, predicted):
    # use the mean squared differences
    return numpy.average((actual - predicted)**2)


def calculate_gradient(x, actual, predicted):
    """
    This calculates gradient for a linear regession 
    using the Mean Squared Error cost function
    """
    # The partial derivatives of MSE are as follows
    # You don't need to be able to do this just yet but
    # it is important to note these give you the two gradients
    # that we need to train our model
    error = predicted - actual
    grad_intercept = numpy.mean(error) * 2
    grad_slope = (x * error).mean() * 2

    return grad_intercept, grad_slope



def gradient_descent(x, y, learning_rate, number_of_iterations):
    """
    Performs gradient descent for a two-parameter function. 

    learning_rate: Larger numbers follow the gradient more aggressively
    number_of_iterations: The maximum number of iterations to perform
    """

    model = MyModel()
    # set the initial parameter guess to 0
    model.intercept = 0
    model.slope = 0
    model.cost_history = []

    last_cost = float('inf')

    for i in range(number_of_iterations):
        # Calculate the predicted values
        predicted = model.predict(x)

        # == OPTIMISER ===
        # Calculate the gradient
        grad_intercept, grad_slope = calculate_gradient(x, y, predicted)
        # Upx the estimation of the line
        model.slope -= learning_rate * grad_slope
        model.intercept -= learning_rate * grad_intercept

        estimate = model.predict(x)
        cost = cost_function(y, estimate)
        # Upx the history of costs
        model.cost_history.append(cost_function(y, estimate))    

        # Print the current estimation and cost every 100 iterations
        if( i % 100 == 0):     
              
            print("Iteration", i, " Current estimate:", model.get_summary(), f"Cost: {cost}")

            if (cost + 0.001) >= last_cost:
                print("Model training complete after",i, "iterations")
                break
            last_cost = cost

    if i == (number_of_iterations - 1):
        print("Maximum number of iterations reached. Stopping training")

    # # Print the final model
    # print(f"Final estimate:", model.get_summary())
    return model
