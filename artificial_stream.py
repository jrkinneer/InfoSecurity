import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, accuracy_score
import pandas as pd

#import data
raw_data = pd.read_csv('creditcard.csv', delimiter=',', on_bad_lines='skip')

#process and split data
y = np.array(raw_data['Class'])

raw_data = raw_data.drop('Class', axis=1)

feature_names = list(raw_data.columns)
features = np.array(raw_data)

#we want the positive or fraudulent class
target_class = 1  

# Select data points belonging to the chosen class
class_data = features[y == target_class]

# Calculate mean and standard deviation for each feature within the class
means = np.mean(class_data, axis=0)
std_devs = np.std(class_data, axis=0)

# Number of artificial test data points to generate
num_samples = 1000

# Generate artificial test data based on normal distribution
artificial_data = np.random.normal(loc=means, scale=std_devs, size=(num_samples, len(means)))

#split original data
train_X, _, train_y, _ = train_test_split(features, y, test_size=.05, random_state=42)
        
rf = RandomForestClassifier(n_estimators=5, random_state=42, max_features=20)

rf.fit(train_X, train_y)
        
predictions = rf.predict(artificial_data)

actual = np.ones(1000)

print("accuracy on artificial data = ", accuracy_score(actual, predictions))
print("balanced accuracy on artificial data = ", balanced_accuracy_score(actual, predictions))