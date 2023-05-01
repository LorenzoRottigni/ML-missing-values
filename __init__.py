""" ML missing values

Substitute missing values not present in the inital dataset with the most frequent
"""

import numpy as np
from sklearn.compose import ColumnTransformer
# Algorithm to substitute missing values of dataset
from sklearn.impute import SimpleImputer
# IterativeInputer is a more complex algorithm to substitute missing values with estimations

X = [
    [20, np.nan],
    [np.nan, 'm'],
    [30, 'f'],
    [35, 'f'],
    [np.nan, np.nan],
]


"""
# Generic transformer appliable to both columns
[
    # Generic input to fill both A & B columns
    'inputer',
    # Apply SimpleImputer algorithm to dataset transformer
    SimpleImputer(
        # Use most frequent occurences to fill missing values because dataset also contains string datatypes.
        # Numeric values strategies: "mean" => average and "median" => middle value
        # String values strategies: "most_frequent" => most frequent value
        # "contastant" => fill with constant "fill_value" parameter
        strategy='most_frequent',
        # Define the format of expected missing values
        missing_values=np.nan
    ),
    # Apply the algorithm to both columns A & B
    [0, 1]
],
"""
transfomers = [
    [
        # Specific age imputer to handle missing values in age feature
        'age_imputer',
        # Apply SimpleImputer algorithm to dataset transformer with median strategy
        SimpleImputer(strategy='mean'),
        # Apply the algoritmh only to first(age) column
        [0]
    ],
    [
        # Specific gender imputer to handle missing values in gender feature
        'gender_inputer',
        # Apply SimpleImputer algorithm to dataset transformer with most frequent strategy
        SimpleImputer(strategy='constant', fill_value='n.d.'),
        # Apply the algoritmh only to second(gender) column
        [1]
    ]
]

ct = ColumnTransformer(transformers=transfomers)

# Overwrite the inital dataset with transformed one
X = ct.fit_transform(X)

print(X)
"""
The most frequent strategy is procedural so it will substitute the missing values
with the most frequent values found until the assignment time.

[[20 'f']
 [20 'm'] # Substitued np.nan with 20 because is the most frequent value untill now
 [30 'f']
 [25 'f']
 [20 'f']] # Substituted both with most frequent values 20 & f
"""

