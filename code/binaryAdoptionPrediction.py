import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import make_scorer

# Load dataset
df = pd.read_csv('../data/Processed_PetFinder_dataset.csv')

# Create the new binary target (1 for adoption, 0 for not adopted)
df['AdoptionBinary'] = df['AdoptionSpeed'].apply(lambda x: 1 if x != 4 else 0)

# Prepare features (X) and target (y)
X = df.drop(['AdoptionSpeed', 'AdoptionBinary'], axis=1)
X = pd.get_dummies(X, drop_first=True)  # One-hot encode categorical features
y = df['AdoptionBinary']

# Load the list of numerical variables from the file
with open('../data/numerical_vars.txt', 'r') as file:
    numerical_vars = file.read().splitlines()

# Filter only the numerical variables that exist in X
numerical_vars = [col for col in numerical_vars if col in X.columns]
X = X[numerical_vars]

# Define the classifiers
classifiers = {
    'RandomForest': RandomForestClassifier(random_state=44),
    'DecisionTree': DecisionTreeClassifier(random_state=44),
    'LogisticRegression': LogisticRegression(random_state=44),
    'NaiveBayes': GaussianNB(),
    'KNeighbors': KNeighborsClassifier(),
    'SVC': SVC(random_state=44),
    'Ensemble': VotingClassifier(estimators=[
        ('rf', RandomForestClassifier(random_state=44)),
        ('dt', DecisionTreeClassifier(random_state=44)),
        ('knn', KNeighborsClassifier())
    ], voting='hard')
}

# Define k-fold cross-validation
kfold = KFold(n_splits=5, shuffle=True, random_state=44)

# Initialize dictionary to store results
results = {}

# Cross-validation and evaluation
for name, clf in classifiers.items():
    print(f"\nEvaluating {name}...")
    
    # Perform cross-validation
    cv_results = cross_val_score(clf, X, y, cv=kfold, scoring='accuracy')
    results[name] = {
        'cv_results': cv_results,
        'mean_accuracy': cv_results.mean(),
        'std_accuracy': cv_results.std()
    }
    
    # Print cross-validation scores
    print(f"Cross-Validation Accuracy Scores: {cv_results}")
    print(f"Mean Accuracy: {cv_results.mean():.4f}")
    print(f"Standard Deviation: {cv_results.std():.4f}")
    
    # Fit the model and predict
    clf.fit(X, y)
    y_pred = clf.predict(X)
    
    # Confusion Matrix
    cm = confusion_matrix(y, y_pred)
    print(f"Confusion Matrix for {name}:\n{cm}")
    
    # Classification Report
    report = classification_report(y, y_pred)
    print(f"Classification Report for {name}:\n{report}")

# Display a summary of all models
print("\nSummary of Model Performance:\n")
summary_df = pd.DataFrame({
    'Model': list(results.keys()),
    'Mean Accuracy': [r['mean_accuracy'] for r in results.values()],
    'Standard Deviation': [r['std_accuracy'] for r in results.values()]
})

# Sort the models by Mean Accuracy
summary_df = summary_df.sort_values(by='Mean Accuracy', ascending=False)

# Print the summary table
print(summary_df)

# Find the best classifier based on mean accuracy
best_classifier = max(results, key=lambda k: results[k]['mean_accuracy'])
print(f"\nBest classifier based on mean accuracy: {best_classifier}")
