'''
This is not intended as learning material.

This generates the data we use for learning module 2c
'''

import numpy as np
import pandas
from scipy.sparse import data
import graphing
from scipy.stats import skewnorm
from sklearn.metrics import accuracy_score

tree_roughness = []
tree_color = []
tree_size = []
tree_motion = []

n_trees = 800
n_rocks = 800
n_animals = 200
n_hikers = 400

np.random.seed(1234567)
rng = np.random.default_rng()

def skewed_normal(skew, n_samples):
    return skewnorm.rvs(skew, size=n_samples)

# Generate trees
tree_color = rng.choice(["green", "brown", "white", "grey", "black"], n_trees, p=[0.2, 0.35, 0.25, 0.15, 0.05])
tree_roughness = (rng.standard_normal(n_trees) * 10 + 30)/30
tree_motion = skewed_normal(0.4, n_trees) * 0.5
tree_size = rng.standard_normal(n_trees) * 6 + (tree_roughness + 1) / (tree_motion + 1) * 20 # size is proportional to roughness as large trees are rough. Small trees also blow more in the wind
tree_labels = ["tree"] * n_trees


# Generate rocks
rock_color = rng.choice(["green", "brown", "white", "grey", "black"], n_rocks, p=[0.1, 0.25, 0.35, 0.25, 0.05])
rock_roughness = (rng.standard_normal(n_rocks) * 10 + 30)/30
rock_motion = rng.uniform(0, 0.05, size=n_rocks) # measurement error
rock_size = rng.standard_normal(n_rocks) + 1.75
rock_labels = ["rock"] * n_rocks


# Generate hikers
hike_color = rng.choice(["green", "brown", "white", "red", "blue", "grey", "orange", "black"], n_hikers, p=[0.2, 0.25, 0.01, 0.04, 0.1, 0.05, 0.05, 0.3])
hike_roughness = (rng.standard_normal(n_hikers) * 2 + 5)/5
hike_motion = skewed_normal(0.4, n_hikers) * 1.5 + 2
hike_size = rng.standard_normal(n_hikers) * 0.3 + 1.75
hike_labels = ["hiker"] * n_hikers


# Generate animals
animal_color = rng.choice(["brown", "white", "red", "black"], n_animals, p=[0.3, 0.25, 0.05, 0.4])
animal_roughness = (rng.standard_normal(n_animals) * 2 + 4.7)/5 # similar to people
animal_motion = skewed_normal(0.4, n_animals) * 3 + 2 # more variable than people
animal_size = rng.standard_normal(n_animals) * 0.4 + 1.6
animal_labels = ["animal"] * n_animals

# Concat
color = np.hstack([tree_color, rock_color, hike_color, animal_color])
size = np.hstack([tree_size, rock_size, hike_size, animal_size])
roughness = np.hstack([tree_roughness, rock_roughness, hike_roughness, animal_roughness])
motion = np.hstack([tree_motion, rock_motion, hike_motion, animal_motion])
label = np.hstack([tree_labels, rock_labels, hike_labels, animal_labels])

# Prevent negative values
roughness[roughness< 0] = 0
motion[motion < 0] = 0
size[size < 0.2] = 0.2

d = pandas.DataFrame(dict(size=size, roughness=roughness, color=color, motion=motion, label=label))

d.to_csv("Data/snow_objects.csv", sep="\t", index=False)

# View as graphs
if True:
    graphing.multiple_histogram(d, label_x="size",  label_group="label", show=True)
    graphing.multiple_histogram(d, label_x="roughness",  label_group="label", show=True)
    graphing.multiple_histogram(d, label_x="motion",  label_group="label", show=True)
    graphing.multiple_histogram(d, label_x="color",  label_group="label", show=True)



# Example of data fitting
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix

# Import the data from the .csv file

dataset = pandas.read_csv('Data/snow_objects.csv', delimiter="\t")

#Let's have a look at the data and the relationship we are going to model
print(dataset.head())

# Convert colors to one-hot
dataset = dataset.join(pandas.get_dummies(dataset.color))
dataset = dataset.drop("color", axis=1)

# Recheck
print(dataset.head())


# Split the dataset in an 75/25 train/test ratio. 
train, test = train_test_split(dataset, test_size=0.3, random_state=1, shuffle=True)



features = [c for c in dataset.columns if c != "label"]

print(features)

def assess_performance(model):
    '''
    Asesses model performance

    '''

    def sub(source):
        actual = source["label"]
        predictions = model.predict(source[features])
        correct = actual == predictions

        tp = np.sum(correct & actual) / len(actual)
        tn = np.sum(correct & np.logical_not(actual)) / len(actual)
        fp = np.sum(np.logical_not(correct) & actual) / len(actual)
        fn = np.sum(np.logical_not(correct) & np.logical_not(actual)) / len(actual)

        print("tp", tp)
        print("tn", tn)
        print("fp", fp)
        print("fn", fn)
        print("Accuracy:", accuracy_score(actual, predictions))
        sensitivity = tp/(fn+tp)




        np.set_printoptions(precision=2)

        # Plot non-normalized confusion matrix
        titles_options = [("Confusion matrix, without normalization", None),
                        ("Normalized confusion matrix", 'true')]
        for title, normalize in titles_options:
            disp = plot_confusion_matrix(model, source[features], source.label,
            cmap=plt.cm.Blues,
            normalize=normalize)
            disp.ax_.set_title(title)

            print(title)
            print(disp.confusion_matrix)

        plt.show()




        return sensitivity

    print("Train")
    train_sensitivity = sub(train)

    print("Test")
    test_sensitivity = sub(test)
    return train_sensitivity, test_sensitivity



# Create the model
random_forest = RandomForestClassifier(n_estimators=1, random_state=1, verbose=False)

# Train the model
random_forest.fit(train[features], train.label)

# Assess performance
train_sensitivity, test_sensitivity = assess_performance(random_forest)

print("Train Sensitivity:", train_sensitivity)
print("Test Sensitivity:", test_sensitivity)