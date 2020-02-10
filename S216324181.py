import pandas as pd
import numpy as np
# ----------------------------------------------
# Other imports here
# ----------------------------------------------
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, MinMaxScaler

# Line must stay the same, except for your student number
# The STUDENTNUMBER.csv file must be your version of the weather.csv file
# You may pre-process the .csv file externally using tools such as Excel, C#, etc if you choose to.
# You are not allowed decrease the number of rows in the dataset
# You may modify the number of columns externally, i.e. make them more or fewer as required
dataset = pd.read_csv("./Data/Assignment2DataSets/216324181.csv")
# Line must stay the same
print(len(dataset))
# -------------------------------------------------------------
# If you want to do any Python-based pre-processing of the dataset, do it here.
# You are not allowed decrease the number of rows in the dataset
# You may modify the number of columns externally, i.e. make them more or fewer as required
# If you want to pre-process the file in Python, look into things such as the following:
# - How to handle NaN
# - Label Encoding
# - One hot encoding
# -------------------------------------------------------------
# Dataset info before pre-processing
print(dataset.info())
# Handling NaN

# Replaced with mean
# Data with integer values
dataset['MinTemp'] = dataset['MinTemp'].fillna(dataset['MinTemp'].mean())
dataset['MaxTemp'] = dataset['MaxTemp'].fillna(dataset['MaxTemp'].mean())
dataset['Rainfall'] = dataset['Rainfall'].fillna(dataset['Rainfall'].mean())
dataset['Evaporation'] = dataset['Evaporation'].fillna(dataset['Evaporation'].mean())
dataset['Sunshine'] = dataset['Sunshine'].fillna(dataset['Sunshine'].mean())
dataset['WindGustSpeed'] = dataset['WindGustSpeed'].fillna(dataset['WindGustSpeed'].mean())
dataset['WindSpeed9am'] = dataset['WindSpeed9am'].fillna(dataset['WindSpeed9am'].mean())
dataset['WindSpeed3pm'] = dataset['WindSpeed3pm'].fillna(dataset['WindSpeed3pm'].mean())
dataset['Humidity9am'] = dataset['Humidity9am'].fillna(dataset['Humidity9am'].mean())
dataset['Humidity3pm'] = dataset['Humidity3pm'].fillna(dataset['Humidity3pm'].mean())
dataset['Pressure9am'] = dataset['Pressure9am'].fillna(dataset['Pressure9am'].mean())
dataset['Pressure3pm'] = dataset['Pressure3pm'].fillna(dataset['Pressure3pm'].mean())
dataset['Cloud9am'] = dataset['Cloud9am'].fillna(dataset['Cloud9am'].mean())
dataset['Cloud3pm'] = dataset['Cloud3pm'].fillna(dataset['Cloud3pm'].mean())
dataset['Temp9am'] = dataset['Temp9am'].fillna(dataset['Temp9am'].mean())
dataset['Temp3pm'] = dataset['Temp3pm'].fillna(dataset['Temp3pm'].mean())

# Data with string values
dataset['RainToday'] = dataset['RainToday'].fillna(0)

# Pre processing for Yes/No columns
dataset['RainTomorrow'].replace({'No': 0, 'Yes': 1}, inplace=True)
dataset['RainToday'].replace({'No': 0, 'Yes': 1}, inplace=True)

# Converting Categorical Data to integers
categoricalColumns = ['Location', 'WindGustDir', 'WindDir9am', 'WindDir3pm']
dataset = pd.get_dummies(dataset, columns=categoricalColumns, dummy_na=True)

# Label encoding dates
le = LabelEncoder()
dataset['Date'] = le.fit_transform(dataset['Date'])
# -------------------------------------------------------------
X = dataset.drop('RainTomorrow', axis=1)  # Line must stay the same
X = X.drop('RISK_MM', axis=1)  # Line must stay the same
# You may decide which other columns to drop, e.g.
# X = X.drop("ColumnName", axis=1)
# -------------------------------------------------------------
y = dataset['RainTomorrow']  # Line must stay the same
# The splitting must stay the same as below, but you must set the random_state equal to your student number
# Example:  X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.20, random_state=STUDENTNUMBER)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=216324181)

# You may select any tree-based classifier with any settings you want to try
# The classifier must be named classifier (as below).  The rest of the script depends on it
classifier = RandomForestClassifier()  # Chang to the type of classifier you want to use

classifier.fit(X_train, y_train)  # Line must stay the same

y_pred = classifier.predict(X_test)  # Line must stay the same

# All the statements below must stay the same
# Your final mark will be determined as follows:
# 50% of your mark will be determined by your model's weighted average f1 score
# 50% of your mark will be determined by your model's accuracy
# Both values will be rounded to 3 decimal digits
# Your final combined result will be rounded up
# Example: (f1: 0.803 * 50) + (accuracy:  0.873 * 50) = 40.15 + 43.65 = 83.8 => 84
# 10% will be deducted if your dataset is not named according to your student number and does not use the correct path
# 10% will be deducted if the number of rows in the dataset are decreased
# 10% will be deducted if you did not use your student number as the random_state
print("Report:\n", classification_report(y_test, y_pred))
print("Accuracy:  ", accuracy_score(y_test, y_pred))
