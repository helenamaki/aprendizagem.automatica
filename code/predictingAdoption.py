import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, make_scorer

# Load dataset
df = pd.read_csv('./data/Processed_PetFinder_dataset.csv')

# Create the new binary target
df['AdoptionBinary'] = df['AdoptionSpeed'].apply(lambda x: 1 if x != 4 else 0)

# Prepare features (X) and target (y)
X = df.drop(['AdoptionSpeed', 'AdoptionBinary'], axis=1)

# Handle categorical variables if any (e.g., using one-hot encoding)
X = pd.get_dummies(X, drop_first=True)
y = df['AdoptionBinary']

# Define the classifier
rf_classifier = RandomForestClassifier(random_state=44)

# Define k-fold cross-validation
kfold = KFold(n_splits=5, shuffle=True, random_state=44)

# Define a scoring function (accuracy here, but you can customize it)
scoring = make_scorer(accuracy_score)

# Perform cross-validation
cv_results = cross_val_score(clf, X, y, cv=kfold, scoring='accuracy')

# Print results
print(f"Cross-Validation Accuracy Scores: {cv_results}")
print(f"Mean Accuracy: {cv_results.mean():.4f}")
print(f"Standard Deviation: {cv_results.std():.4f}")
