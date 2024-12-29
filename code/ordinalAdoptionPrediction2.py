import pandas as pd
import numpy as np
import json
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, make_scorer, mean_squared_error, cohen_kappa_score
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

# Load dataset
df = pd.read_csv('../data/Processed_PetFinder_dataset.csv')

# Create the ordinal target (AdoptionSpeed: 0, 1, 2, 3, 4)
X = df.drop(['AdoptionSpeed'], axis=1)
X = pd.get_dummies(X, drop_first=True)  # One-hot encode categorical features
y = df['AdoptionSpeed']

# Load the list of numerical variables from the file
with open('../data/numerical_vars.txt', 'r') as file:
    numerical_vars = file.read().splitlines()

# Filter only the numerical variables that exist in X
numerical_vars = [col for col in numerical_vars if col in X.columns]
X = X[numerical_vars]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=44)

# Function to save hyperparameters to JSON file
def save_hyperparameters(model_name, hyperparameters, filename='../data/ordinal/best_hyperparameters.json'):
    try:
        with open(filename, 'r') as f:
            all_params = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        all_params = {}

    all_params[model_name] = hyperparameters

    with open(filename, 'w') as f:
        json.dump(all_params, f, indent=4)

# Function to load hyperparameters from JSON file
def load_hyperparameters(model_name, filename='../data/ordinal/best_hyperparameters.json'):
    try:
        with open(filename, 'r') as f:
            all_params = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None

    return all_params.get(model_name, None)

# Define custom scoring metrics
def quadratic_weighted_kappa(y_true, y_pred):
    return cohen_kappa_score(y_true, y_pred, weights='quadratic')

mse_scorer = make_scorer(mean_squared_error, greater_is_better=False)
qwk_scorer = make_scorer(quadratic_weighted_kappa)

# Define classifiers and parameter grids
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

param_grids = {
    'RandomForest': {
        'n_estimators': [100, 200],
        'max_features': ['auto', 'sqrt'],
        'max_depth': [None, 10],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'bootstrap': [True, False]
    },
    'DecisionTree': {
        'max_depth': [None, 10],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'criterion': ['gini', 'entropy']
    },
    'LogisticRegression': {
        'C': [0.1, 1],
        'solver': ['liblinear'],
        'max_iter': [100, 200]
    },
    'KNeighbors': {
        'n_neighbors': [3, 5],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan']
    },
    'SVC': {
        'C': [0.1, 1],
        'kernel': ['linear', 'rbf'],
        'gamma': ['scale', 'auto']
    }
}

# Hyperparameter tuning and evaluation
results = {
    'Model': [],
    'Mean Accuracy': [],
    'Std Accuracy': [],
    'Mean MSE': [],
    'Mean QWK': []
}

for model_name, clf in classifiers.items():
    if model_name == "NaiveBayes" or model_name == "Ensemble":
        continue

    print(f"\nTuning hyperparameters for {model_name}...")
    best_params = load_hyperparameters(model_name)

    if best_params is None:
        grid_search = GridSearchCV(
            clf, param_grids[model_name],
            scoring={'accuracy': 'accuracy', 'mse': mse_scorer, 'qwk': qwk_scorer},
            refit='qwk',  # Optimize for QWK
            cv=5, n_jobs=-1
        )
        grid_search.fit(X_train, y_train)
        best_params = grid_search.best_params_
        save_hyperparameters(model_name, best_params)
        print(f"Best hyperparameters for {model_name}: {best_params}")
    else:
        print(f"Using saved hyperparameters for {model_name}: {best_params}")
        clf.set_params(**best_params)

    # Evaluate with cross-validation
    print(f"\nEvaluating {model_name}...")
    cv_results_accuracy = cross_val_score(clf, X, y, cv=5, scoring='accuracy', n_jobs=-1)
    cv_results_mse = cross_val_score(clf, X, y, cv=5, scoring=mse_scorer, n_jobs=-1)
    cv_results_qwk = cross_val_score(clf, X, y, cv=5, scoring=qwk_scorer, n_jobs=-1)

    results['Model'].append(model_name)
    results['Mean Accuracy'].append(cv_results_accuracy.mean())
    results['Std Accuracy'].append(cv_results_accuracy.std())
    results['Mean MSE'].append(-cv_results_mse.mean())  # Negate MSE as it was minimized
    results['Mean QWK'].append(cv_results_qwk.mean())

    print(f"Mean Accuracy: {cv_results_accuracy.mean():.4f}")
    print(f"Std Accuracy: {cv_results_accuracy.std():.4f}")
    print(f"Mean MSE: {-cv_results_mse.mean():.4f}")
    print(f"Mean QWK: {cv_results_qwk.mean():.4f}")

# Convert results to DataFrame for easy visualization
summary_df = pd.DataFrame(results)
summary_df = summary_df.sort_values(by='Mean QWK', ascending=False)
print("\nSummary of Model Performance:\n")
print(summary_df)
