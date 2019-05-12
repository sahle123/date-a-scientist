#smokers_2_Regression.py
#
# Using multi-linear regression and K-Nearest Neighbors to predict if
# a person is a smoker based on their age, income, and level of education.
#
# Sahle "Nomad" Alturaigi
# LU: 05/11/2019

import pandas as pd
import numpy as np
import statistics as stat
import random
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier


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


#
# Clean and reorganize data
#
# Separate out smokers and non-smokers
smokers_df = df[df["smokes"]!="no"]
non_smokers_df = df[df["smokes"]=="no"]

temp_smokers_df = pd.DataFrame(data = smokers_df[["body_type", "age"]])
temp_non_smokers_df = pd.DataFrame(data = non_smokers_df[["body_type", "age"]])

# Add labels
temp_smokers_df["is_smoker"] = [1]*(len(temp_smokers_df))
temp_non_smokers_df["is_smoker"] = [0]*(len(temp_non_smokers_df))


# Combine into working data frame and drop all NaNs
working_df = pd.concat([temp_smokers_df, temp_non_smokers_df.sample(n=len(temp_smokers_df))], axis=0, sort=False)
working_df.reset_index(inplace=True)
working_df.dropna(inplace=True, axis=0)

#
# Get age and income and normalize them.
#
X = working_df[["age"]]
y = working_df[["is_smoker"]]

# Encode body_type data into the following dictionary
body_type_mapping = {"average": 0, "fit": 1, "athletic": 2, "thin": 3,\
                     "curvy": 4, "a little extra": 5, "skinny": 6, \
                     "full figured": 7, "overweight": 8, "jacked": 9,\
                     "used up": 10, "rather not say": 11}

X = X.assign(body_type_enc=(working_df["body_type"].map(body_type_mapping)))

# Split data
x_train, x_test, y_train, y_test = train_test_split(X, y, train_size = 0.8, test_size = 0.2, random_state=1)

# Feature scaling
# N.B. no need for feature scaling on labels since they are already binary.
#sc_X = StandardScaler()
#x_train = sc_X.fit_transform(x_train[["age", "income"]].astype("float64"))
#x_test = sc_X.transform(x_test[["age", "income"]].astype("float64"))


#
# Instantiate, run regressor, predictions
#
regressor = LinearRegression()

model = regressor.fit(x_train, y_train)

y_predict = model.predict(x_test)

print("Multi-linear regression")
print("Train score:")
print(model.score(x_train, y_train))
print("Test score:")
print(model.score(x_test, y_test))
print("------------------")



#
# Using K-nearest neighbors
#
print("K-Neartest neighbors")

best_k = 0
best_score = 0
accuracies = []
for k in range(1, 201):  
    classifier = KNeighborsClassifier(n_neighbors=k)

    # Fit classifier with data
    classifier.fit(x_train, np.ravel(y_train))

    # Score classifier against validation data
    score = classifier.score(x_test, np.ravel(y_test))
    accuracies.append(score)
    print(str(score*100)[:5] + "% Accurate for k: " + str(k))

    # Check if better k value was found
    if best_score < score:
        best_score = score
        best_k = k

    # Find best k value
    print("Best k value: %i" % best_k)

    # Generate x-axis values
    k_list = range(1, 201)


# Plot k value accuracies
plt.figure()
plt.xlabel("K values")
plt.ylabel("Validation Accuracy (%)")
plt.title("Smoker Classifier Accuracy")
plt.plot(k_list, accuracies, color="orange")
plt.savefig("KNN classifier")
plt.show()