#smokers_2_SVC.py
#
# Using Support Vector Classifier to see how what phrases are considered "smoker" or "non-smoker" phrases
# with Scoring.
#
# Sahle "Nomad" Alturaigi
# LU: 05/11/2019

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC



# ----------------------------------------------------------------------
# Globals
# Create your df here:
df = pd.read_csv("profiles.csv")



# ----------------------------------------------------------------------
# Prototypes
# 
# Returns F1 score
def classify_and_score(C_in, training_counts, training_labels,\
    test_counts, test_labels):

    # Using a linear kernel
    classifier = SVC(kernel="linear", C=C_in, probability=True)
    classifier.fit(training_counts, training_labels)

    #print(classifier.predict_proba(test_counts))
    guesses = classifier.predict(test_counts)
    f1 = f1_score(test_labels, guesses)
    
    print("C = " + str(C_in))
    print("Accuracy: " + str(accuracy_score(test_labels, guesses)))
    print("Precision: " + str(precision_score(test_labels, guesses)))
    print("Recall: " + str(recall_score(test_labels, guesses)))
    print("F1 score: " + str(f1))
    print("---------")
    
    return f1

# ----------------------------------------------------------------------

#
# ASSUMPTION: Anyone who smokes, even those trying to quit, are categorized as "smokers".
# Anyone who has listed "no" to smoking is a non-smoker.
#

def main():
    
    #
    # Clean and reorganize data
    #
    # Separate out smokers and non-smokers
    smokers_df = df[df["smokes"]!="no"]
    non_smokers_df = df[df["smokes"]=="no"]

    # Store only essay6 and remove all empty essay entries
    smoker_essay_set = pd.DataFrame(data = smokers_df["essay6"].dropna())
    non_smoker_essay_set = pd.DataFrame(data = non_smokers_df["essay6"].dropna())


    # Add labels
    smoker_essay_set["is_smoker"] = [1]*(len(smoker_essay_set))
    non_smoker_essay_set["is_smoker"] = [0]*(len(non_smoker_essay_set))


    # Combine tables into a working set dataframe
    working_df = pd.concat([smoker_essay_set, non_smoker_essay_set.sample(n=len(smoker_essay_set))], axis=0, sort=False)
    working_df.reset_index(inplace=True)            

    # Split data here into training and test sets.
    training_data, test_data, training_labels, test_labels = \
        train_test_split(working_df["essay6"], working_df["is_smoker"], train_size=0.8, test_size=0.2)

    # Changing series to lists
    training_data = training_data.values.tolist()[:2000]
    test_data = test_data.values.tolist()[:2000]
    training_labels = training_labels.values.tolist()[:2000]
    test_labels = test_labels.values.tolist()[:2000]


    #
    # Count Vectorizer
    #
    counter = CountVectorizer()

    # DEV-NOTE: computationally heavy step
    counter.fit(working_df["essay6"].values.tolist())

    training_counts = counter.transform(training_data)
    test_counts = counter.transform(test_data)


    #
    # Support Vector Classifier
    #
    return classify_and_score(1, training_counts, training_labels, test_counts, test_labels)


if __name__ == "__main__":
    i = 0 # Control variable
    e = 10 # How many iterations to run
    
    print("""Note that due to the large size of our data set, the SVC keeps hanging on my machine. As a workaround,
    I take a randomized set of 2,000 words to train on. Since this is an order of magnitude smaller, I've had the
    model run " + str(e) + " times over and took the average f1 score from their to get a rough idea how good the
    classifer might've been had I been able to operate under the whole dataset.""")
    
    f1_total = 0
    while i < e:
        f1_total += main()
        i += 1
    print("Average F1 score after "+str(e) + " iterations is: " + str(f1_total/e))