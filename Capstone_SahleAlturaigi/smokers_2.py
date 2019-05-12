#smokers_2.py
#
# Using Naive Bayes' Classifier to see how what phrases are considered "smoker" or "non-smoker" phrases
#
#
# Sahle "Nomad" Alturaigi
# LU: 05/09/2019

import pandas as pd
import numpy as np
import statistics as stat
import random
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB



# ----------------------------------------------------------------------
# Globals
# Create your df here:
df = pd.read_csv("profiles.csv")

#
# Phrase you want to use. Change this variable to whatever string you like.
#
LE_TEST_TEXT = "You got a lighter?"


# Number of times to run
N = 3 



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

    n = 0
    while n < N:
        # Add labels
        smoker_essay_set["is_smoker"] = [1]*(len(smoker_essay_set))
        non_smoker_essay_set["is_smoker"] = [0]*(len(non_smoker_essay_set))
        

        # Combine tables into a working set dataframe
        working_df = pd.concat([smoker_essay_set, non_smoker_essay_set.sample(n=len(smoker_essay_set))], axis=0, sort=False)
        working_df.reset_index(inplace=True)            


        #
        # Count Vectorizer
        #
        counter = CountVectorizer()

        # DEV-NOTE: computationally heavy step
        counter.fit(working_df["essay6"].values.tolist())

        training_counts = counter.transform(working_df["essay6"])



        #
        # Naive Bayes' Classifier
        #
        classifier = MultinomialNB()
        classifier.fit(training_counts, working_df["is_smoker"].values.tolist())

        le_test_counts = counter.transform([LE_TEST_TEXT])

        print("Word: " + LE_TEST_TEXT)
        print(classifier.predict_proba(le_test_counts))
        translate_result(classifier.predict(le_test_counts))
        n += 1
        
    print("---------")
    return


if __name__ == "__main__":
    main()
