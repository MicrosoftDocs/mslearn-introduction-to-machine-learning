'''
This code generates the dataset used for the avalanche generation dataset
This code is absolutely, and in no way, shape, or form, designed to mimic any real process
The resulting dataset should, in no way, be considered to be real or used for any
real avalanche prediction. It is pure fiction, in order to generate learning material
for teaching machine learning techniques 
'''
import numpy as np
import numpy.random as rng
import pandas
import statsmodels.formula.api as smf
import graphing
import sklearn.model_selection
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


weak_layer_safeties = []

class SnowCondition:
    def __init__(self, surface_hoar:float, fresh_thickness:float, wind:float, weak_layers:int, tracked_out:bool) -> None:
        self.surface_hoar = surface_hoar
        self.fresh_thickness = fresh_thickness
        self.wind = wind
        self.weak_layers = weak_layers
        self.tracked_out = tracked_out

    
    def get_avalanche(self, visitors:int) -> bool:
        '''
        Determine if an avalanche takes place
        '''

        bias = 0.35 # fudge factor to get a fairly even split of avalanche and non-avalanche days
        def logit(val):
            return 1- bias/(1+np.exp(-val)) 

        # convert wind from km/h to a risk
        wind_safety = logit((self.wind-20)*0.5)
        top_safety = logit(self.surface_hoar - 5)
        weak_layer_safety = logit(self.weak_layers - 5)

        visitor_safety = logit((visitors - 5) * (self.fresh_thickness - 5)/2)
        # thickness_safety = 1 - bias / (1+np.exp(-(self.fresh_thickness - 5)/2))
        # visitor_safety = (visitor_safety + thickness_safety)/2

        threshold = wind_safety * top_safety * weak_layer_safety * visitor_safety + rng.standard_normal() * 0.1
        weak_layer_safeties.append(threshold < 0.5)

        return threshold < 0.5


n_samples = int(365.25*3)


rng.seed(1)


snow_cond = []
no_visitors = []
avalanche = []

for i in range(n_samples):
    condition = SnowCondition(
        surface_hoar = rng.randint(0,8) + rng.standard_normal() * 2,  
        fresh_thickness = rng.randint(0,8) + rng.standard_normal() * 2, 
        wind = rng.randint(0,41), #restrict(rng.standard_normal() * 10 + 20,0,80),
        weak_layers = rng.randint(0,11), 
        tracked_out = rng.randint(0,2))

    visitors = rng.randint(10)

    snow_cond.append(condition)
    no_visitors.append(visitors)
    avalanche.append(int(condition.get_avalanche(visitors)))

df = pandas.DataFrame(dict(
    avalanche = avalanche,
    no_visitors = no_visitors,
    surface_hoar = [c.surface_hoar for c in snow_cond],
    fresh_thickness = [c.fresh_thickness for c in snow_cond],
    wind = [c.wind for c in snow_cond],
    weak_layers = [c.weak_layers for c in snow_cond],
    tracked_out = [c.tracked_out for c in snow_cond])
    )

# Patch to avoid negative snow
df.fresh_thickness = df.fresh_thickness - df.fresh_thickness.min()


# preview
print(df.head(20))
print(np.average(weak_layer_safeties))

# Save
df.to_csv("Data/avalanche.csv", sep="\t")

# graph
if False:
    graphing.multiple_histogram(df, 'no_visitors', 'avalanche', show=True)
    graphing.multiple_histogram(df, 'surface_hoar', 'avalanche', show=True)
    graphing.multiple_histogram(df, 'fresh_thickness', 'avalanche', show=True)
    graphing.multiple_histogram(df, 'wind', 'avalanche', show=True)
    graphing.multiple_histogram(df, 'weak_layers', 'avalanche', show=True)
    graphing.multiple_histogram(df, 'tracked_out', 'avalanche', show=True)


train, test = sklearn.model_selection.train_test_split(df, test_size=0.2, random_state=10, shuffle=True)

def truth_table(predictions):
    correct = test.avalanche == predictions

    tp = np.sum(correct & test.avalanche) / test.shape[0]
    tn = np.sum(correct & np.logical_not(test.avalanche)) / test.shape[0]
    fp = np.sum(np.logical_not(correct) & test.avalanche) / test.shape[0]
    fn = np.sum(np.logical_not(correct) & np.logical_not(test.avalanche)) / test.shape[0]

    # print(predictions)
    print("---")
    print(accuracy_score(test.avalanche, predictions))
    print("tp", tp)
    print("tn", tn)
    print("fp", fp)
    print("fn", fn)

# Example logistic model
print("All features")
model = smf.logit("avalanche ~ no_visitors + surface_hoar + fresh_thickness + wind + weak_layers + tracked_out", train).fit()
predictions = model.predict(test) > 0.5
truth_table(predictions)

# Example logistic model
print("Ideal")
model = smf.logit("avalanche ~ no_visitors + fresh_thickness + no_visitors * fresh_thickness + surface_hoar + wind + weak_layers", train).fit()
predictions = model.predict(test) > 0.5
truth_table(predictions)
print(model.summary())

# Simple model 
print("Simple")
model = smf.logit("avalanche ~ weak_layers", train).fit()
predictions = model.predict(test) > 0.5
truth_table(predictions)

# Example random forest
scaler = StandardScaler()
features = ["no_visitors", "surface_hoar", "fresh_thickness", "wind", "weak_layers", "tracked_out"]
train[features] = scaler.fit_transform(train[features])
test[features] = scaler.transform(test[features])

clf = RandomForestClassifier(n_estimators=500, random_state=1, verbose=False)

X_train = train[features]
y_train = train["avalanche"]
clf.fit(X_train,y_train)

X_test = test[features]
y_pred = clf.predict(X_test)
print("Random Forest")

truth_table(y_pred)
