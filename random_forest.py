import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score
import pandas as pd
import matplotlib.pyplot as plt

#import data
raw_data = pd.read_csv('creditcard.csv', delimiter=',', on_bad_lines='skip')

#process and split data
y = np.array(raw_data['Class'])

raw_data = raw_data.drop('Class', axis=1)

feature_names = list(raw_data.columns)
features = np.array(raw_data)

#train
estimators = [3,5,7,9]
num_features = [10, 15, 20, 25, 31]

#test
average_accuracies = []
average_balanced_accuracies = []
for n in estimators:
    accuracy_list = []
    balanced_accuracy_list = []
    precision_list = []
    
    best_balanced_accuracy = (0,0)
    best_accuracy = (0,0)
    
    for f in num_features:
        #randomly split for each iteration
        train_X, test_X, train_y, test_y = train_test_split(features, y, test_size=.2, random_state=42)
        
        rf = RandomForestClassifier(n_estimators=n, random_state=42, max_features=f)
        rf.fit(train_X, train_y)
        
        predictions = rf.predict(test_X)
        
        accuracy = accuracy_score(test_y, predictions)
        balanced_accuracy =balanced_accuracy_score(test_y, predictions)
        precision = precision_score(test_y, predictions) 
         
        accuracy_list.append(accuracy)
        balanced_accuracy_list.append(balanced_accuracy)
        precision_list.append(precision)
        
        
        if accuracy > best_accuracy[0]:
            best_accuracy = (accuracy, f)
        if balanced_accuracy > best_balanced_accuracy[0]:
            best_balanced_accuracy = (balanced_accuracy, f)
        
    print("for n trees = ", n)
    print("best balanced accuracy: ", best_balanced_accuracy[0], " at # features: ", best_balanced_accuracy[1])
    print("best accuracy: ", best_accuracy[0], " at # features: ", best_accuracy[1])
    
    
    plt.plot(num_features, accuracy_list)
    title = "accuracy for n trees n = " + str(n)
    plt.title(title)
    plt.ylabel("accuracy")
    plt.xlabel("max features")
    title = title +".png"
    plt.savefig(title)
    plt.close()
    
    plt.plot(num_features, balanced_accuracy_list)
    title = "balanced accuracy for n trees n = " + str(n)
    plt.title(title)
    plt.ylabel("accuracy")
    plt.xlabel("max features")
    title = title +".png"
    plt.savefig(title)
    plt.close()
    
    plt.plot(num_features, balanced_accuracy_list)
    title = "balanced accuracy for n estimators n = " + str(n)
    plt.title(title)
    plt.ylabel("accuracy")
    plt.xlabel("max features")
    title = title +".png"
    plt.savefig(title)
    plt.close()
    
    plt.plot(num_features, precision_list)
    title = "precision for n estimators n = " + str(n)
    plt.title(title)
    plt.ylabel("accuracy")
    plt.xlabel("max features")
    title = title +".png"
    plt.savefig(title)
    plt.close()
      
    average_accuracies.append(sum(accuracy_list)/len(accuracy_list))
    average_balanced_accuracies.append(sum(balanced_accuracy_list)/len(balanced_accuracy_list))
    

print(average_accuracies)
print(average_balanced_accuracies)  
  
#generate new data

#test on new data