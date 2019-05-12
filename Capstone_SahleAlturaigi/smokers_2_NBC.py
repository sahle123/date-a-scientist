#smokers_2_NBC.py
#
# Using Naive Bayes' Classifier to see how what phrases are considered "smoker" or "non-smoker" phrases.
# with model scoring.
#
# Sahle "Nomad" Alturaigi
# LU: 05/09/2019

import pandas as pd
import numpy as np
import statistics as stat
import random
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB



# ----------------------------------------------------------------------
# Globals
# Create your df here:
df = pd.read_csv("profiles.csv")



# ----------------------------------------------------------------------
# Prototypes
# 

def translate_result(ins):
    if ins == 1:
        print("Is a smoker 🚬")
    elif ins == 0:
        print("Not a smoker 🚭")
    else:
        print("Bad Inputs")
    return


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
    training_data = training_data.values.tolist()
    test_data = test_data.values.tolist()
    training_labels = training_labels.values.tolist()
    test_labels = test_labels.values.tolist()


    #
    # Count Vectorizer
    #
    counter = CountVectorizer()

    # DEV-NOTE: computationally heavy step
    counter.fit(working_df["essay6"].values.tolist())

    training_counts = counter.transform(training_data)
    testing_counts = counter.transform(test_data)


    #
    # Naive Bayes' Classifier
    #
    classifier = MultinomialNB()
    classifier.fit(training_counts, training_labels)

    print(classifier.predict_proba(testing_counts))
    guesses = classifier.predict(testing_counts)
    
    print("Accuracy: " + str(accuracy_score(test_labels, guesses)))
    print("Precision: " + str(precision_score(test_labels, guesses)))
    print("Recall: " + str(recall_score(test_labels, guesses)))
    print("F1 score: " + str(f1_score(test_labels, guesses)))
    print("---------")
    return


if __name__ == "__main__":
    main()
