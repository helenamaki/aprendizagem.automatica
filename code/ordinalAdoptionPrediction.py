import os
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, mean_squared_error, cohen_kappa_score
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

# Directory for ordinal data
ordinal_dir = '../data/ordinal/'
os.makedirs(ordinal_dir, exist_ok=True)

# Functions for saving/loading hyperparameters and metrics
def save_hyperparameters(model_name, hyperparameters, filename=f'{ordinal_dir}best_hyperparameters.json'):
    try:
        with open(filename, 'r') as f:
            all_params = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        all_params = {}
    
    all_params[model_name] = hyperparameters
    
    with open(filename, 'w') as f:
        json.dump(all_params, f, indent=4)
    print(f"Hyperparameters for {model_name} saved to {filename}")

def load_hyperparameters(model_name, filename=f'{ordinal_dir}best_hyperparameters.json'):
    try:
        with open(filename, 'r') as f:
            all_params = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None
    
    return all_params.get(model_name, None)

def save_metrics(metrics, filename=f'{ordinal_dir}model_metrics.json'):
    try:
        with open(filename, 'r') as f:
            all_metrics = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        all_metrics = {}
    
    all_metrics.update(metrics)
    
    with open(filename, 'w') as f:
        json.dump(all_metrics, f, indent=4)
    print(f"Metrics saved to {filename}")

def load_metrics(filename=f'{ordinal_dir}model_metrics.json'):
    try:
        with open(filename, 'r') as f:
            all_metrics = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}
    
    return all_metrics

# Load dataset
df = pd.read_csv('../data/Processed_PetFinder_dataset.csv')

# Create ordinal target
df['AdoptionOrdinal'] = df['AdoptionSpeed']  # Use 0 to 4 as is

# Prepare features (X) and target (y)
X = df.drop(['AdoptionSpeed', 'AdoptionOrdinal'], axis=1)
y = df['AdoptionOrdinal']

# Separate categorical and numerical features
categorical_vars = X.select_dtypes(include=['object']).columns.tolist()
numerical_vars = X.select_dtypes(exclude=['object']).columns.tolist()

# Apply one-hot encoding to categorical variables and keep the numerical ones as is
X_categorical = pd.get_dummies(X[categorical_vars], drop_first=True)
X_numerical = X[numerical_vars]

# Combine numerical and categorical features back together
X = pd.concat([X_numerical, X_categorical], axis=1)

# Load the list of numerical variables from the file
with open('../data/numerical_vars.txt', 'r') as file:
    numerical_vars_from_file = file.read().splitlines()

# Filter only the numerical variables that exist in X (so that they match correctly)
numerical_vars_from_file = [col for col in numerical_vars_from_file if col in X.columns]

# Keep only the numerical columns that are present in both X and the file
X = X[numerical_vars_from_file + X_categorical.columns.tolist()]  # Add both numerical and categorical columns

# Define classifiers and parameter grids for tuning
classifiers = {
    'RandomForest': RandomForestClassifier(random_state=44),
    'DecisionTree': DecisionTreeClassifier(random_state=44),
    'LogisticRegression': LogisticRegression(random_state=44, max_iter=1000),
    'NaiveBayes': GaussianNB(),
    'KNeighbors': KNeighborsClassifier(),
    'SVC': SVC(random_state=44, probability=True, gamma='scale'),  # Use 'scale' for gamma
    'Ensemble': VotingClassifier(estimators=[
        ('rf', RandomForestClassifier(random_state=44, n_estimators=200, max_depth=20, min_samples_split=5, min_samples_leaf=2)),
        ('lr', LogisticRegression(random_state=44, max_iter=1000, C=1, solver='liblinear')),
        ('svc', SVC(random_state=44, probability=True, kernel='rbf', C=1, gamma='auto'))
    ], voting='soft')
}

param_grids = {
    'RandomForest': {
        'n_estimators': [100, 200, 500],
        'max_features': ['auto', 'sqrt', 'log2'],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'bootstrap': [True, False]
    },
    'DecisionTree': {
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'criterion': ['gini', 'entropy']
    },
    'LogisticRegression': {
        'C': [0.1, 1, 10],
        'solver': ['liblinear', 'saga']
    },
    'KNeighbors': {
        'n_neighbors': [3, 5, 10],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan']
    },
    'SVC': {
        'C': [0.1, 1, 10],
        'gamma': ['scale', 'auto'],
        'kernel': ['linear', 'rbf', 'poly']
    }
}

# Hyperparameter tuning flag
retune_hyperparameters = True  # Change this flag to False to skip tuning

# Results dictionary for summary
results = {
    'Model': [],
    'Mean Accuracy': [],
    'MSE': [],
    'Quadratic Weighted Kappa': []
}

# Train and evaluate models
for model_name, clf in classifiers.items():
    if model_name == "NaiveBayes":
        print(f"\nSkipping hyperparameter tuning for {model_name}")
        continue
    
    # Hyperparameter tuning for models other than SVC
    if model_name != 'SVC' and retune_hyperparameters and model_name in param_grids:
        print(f"\nTuning hyperparameters for {model_name}...")
        best_params = load_hyperparameters(model_name)
        
        if best_params is None:
            print(f"No saved hyperparameters found for {model_name}, starting grid search...")
            grid_search = GridSearchCV(clf, param_grids[model_name], cv=5, scoring='accuracy', n_jobs=-1)
            grid_search.fit(X, y)
            best_params = grid_search.best_params_
            save_hyperparameters(model_name, best_params)
        else:
            print(f"Using saved hyperparameters for {model_name}: {best_params}")
        
        clf.set_params(**best_params)
    
    # Cross-validation and evaluation
    print(f"\nEvaluating {model_name}...")
    clf.fit(X, y)
    y_pred = clf.predict(X)

    # Metrics
    mse = mean_squared_error(y, y_pred)
    qwk = cohen_kappa_score(y, y_pred, weights='quadratic')
    cv_accuracy = cross_val_score(clf, X, y, cv=5, scoring='accuracy').mean()

    print(f"Confusion Matrix for {model_name}:\n{confusion_matrix(y, y_pred)}")
    print(f"Classification Report for {model_name}:\n{classification_report(y, y_pred)}")
    print(f"Mean Accuracy: {cv_accuracy:.4f}")
    print(f"MSE: {mse:.4f}")
    print(f"Quadratic Weighted Kappa: {qwk:.4f}")

    # Save results
    results['Model'].append(model_name)
    results['Mean Accuracy'].append(cv_accuracy)
    results['MSE'].append(mse)
    results['Quadratic Weighted Kappa'].append(qwk)

# Save results to JSON
save_metrics(results)

# Convert results to DataFrame for display
summary_df = pd.DataFrame(results).sort_values(by='Quadratic Weighted Kappa', ascending=False)
print("\nSummary of Model Performance:\n")
print(summary_df)
