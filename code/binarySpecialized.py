import pandas as pd
import numpy as np
import json
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import multiprocessing
import os

# Load dataset
df = pd.read_csv('../data/Processed_PetFinder_dataset.csv')

# Create the new binary target (1 for adoption, 0 for not adopted)
df['AdoptionBinary'] = df['AdoptionSpeed'].apply(lambda x: 1 if x != 4 else 0)

# Load the list of numerical variables from the file
with open('../data/numerical_vars.txt', 'r') as file:
    numerical_vars = file.read().splitlines()

# Define classifiers and parameter grids for tuning
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

# Define parameter grids for tuning
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
        'solver': ['liblinear', 'saga'],
        'max_iter': [100, 200]
    },
    'KNeighbors': {
        'n_neighbors': [3, 5, 10],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan']
    },
    'SVC': {
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'rbf'],
        'gamma': ['scale', 'auto']
    }
}

# Function to save hyperparameters to JSON file
def save_hyperparameters(model_name, hyperparameters, pet_type, filename='../data/'):
    folder = 'binaryDogs' if pet_type == 1 else 'binaryCats'
    folder_path = os.path.join(filename, folder)
    
    # Create the directory if it doesn't exist
    os.makedirs(folder_path, exist_ok=True)

    # Set the path for the JSON file
    file_path = os.path.join(folder_path, 'best_hyperparameters.json')
    
    # Load the existing data, if any
    try:
        with open(file_path, 'r') as f:
            all_params = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        all_params = {}

    # Add the hyperparameters for the current model
    all_params[model_name] = hyperparameters
    
    # Save the updated hyperparameters to the file
    with open(file_path, 'w') as f:
        json.dump(all_params, f, indent=4)

# Function to load hyperparameters from JSON file
def load_hyperparameters(model_name, pet_type, filename='../data/'):
    folder = 'binaryDogs' if pet_type == 1 else 'binaryCats'
    folder_path = os.path.join(filename, folder)
    
    # Set the path for the JSON file
    file_path = os.path.join(folder_path, 'best_hyperparameters.json')
    
    # Load the existing data, if any
    try:
        with open(file_path, 'r') as f:
            all_params = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None

    # Return the hyperparameters for the given model
    return all_params.get(model_name, None)

# Filter dataset for dogs (Type = 1) and cats (Type = 2)
df_dogs = df[df['Type'] == 1].copy()
df_cats = df[df['Type'] == 2].copy()

# Function to prepare features and target
def prepare_data(df_subset):
    # Prepare features (X) and target (y)
    X = df_subset.drop(['AdoptionSpeed', 'AdoptionBinary', 'Type'], axis=1)  # Drop 'Type' column as it's not a feature
    X = pd.get_dummies(X, drop_first=True)  # One-hot encode categorical features
    y = df_subset['AdoptionBinary']
    
    # Filter only the numerical variables that exist in X
    numerical_vars = [col for col in numerical_vars if col in X.columns]
    X = X[numerical_vars]
    
    return X, y

# Hyperparameter tuning flag
retune_hyperparameters = True  # Change this flag to False to skip tuning

# Function to train and evaluate models for both dogs and cats
def evaluate_models(X, y, classifiers, param_grids, retune_hyperparameters, pet_type):
    results = {
        'Model': [],
        'Mean Accuracy': [],
        'Std Accuracy': []
    }

    for model_name, clf in classifiers.items():
        if model_name == "NaiveBayes" or model_name == "Ensemble":
            continue
        # Check if we need to retune
        if retune_hyperparameters:
            print(f"\nTuning hyperparameters for {model_name}...")

            # Check if hyperparameters are already saved
            best_params = load_hyperparameters(model_name, pet_type)

            if best_params is None:
                print(f"No saved hyperparameters found for {model_name}, starting grid search...")

                # Set up GridSearchCV for the current model with parallelization
                grid_search = GridSearchCV(clf, param_grids[model_name], cv=5, scoring='accuracy', n_jobs=-1)  # n_jobs=-1 for parallelization
                grid_search.fit(X, y)

                # Get the best parameters and model
                best_params = grid_search.best_params_

                # Save the best hyperparameters
                save_hyperparameters(model_name, best_params, pet_type)
                print(f"Best hyperparameters for {model_name}: {best_params}")
            else:
                print(f"Using saved hyperparameters for {model_name}: {best_params}")
            
            # Create the model with the best parameters
            clf.set_params(**best_params)
        else:
            print(f"Skipping tuning for {model_name}, using default parameters.")
        
        # Perform cross-validation with parallelization
        print(f"\nEvaluating {model_name}...")
        cv_results = cross_val_score(clf, X, y, cv=5, scoring='accuracy', n_jobs=-1)  # n_jobs=-1 for parallelization
        print(f"Cross-Validation Accuracy Scores: {cv_results}")
        print(f"Mean Accuracy: {cv_results.mean():.4f}")
        print(f"Standard Deviation: {cv_results.std():.4f}")
        
        # Fit the model and predict
        clf.fit(X, y)
        y_pred = clf.predict(X)
        
        # Confusion Matrix
        cm = confusion_matrix(y, y_pred)
        print(f"Confusion Matrix for {model_name}:\n{cm}")
        
        # Classification Report
        report = classification_report(y, y_pred)
        print(f"Classification Report for {model_name}:\n{report}")
        
        # Add cross-validation results to summary
        results['Model'].append(model_name)
        results['Mean Accuracy'].append(cv_results.mean())
        results['Std Accuracy'].append(cv_results.std())

    # Convert results to DataFrame for easy visualization
    summary_df = pd.DataFrame(results)
    summary_df = summary_df.sort_values(by='Mean Accuracy', ascending=False)
    print("\nSummary of Model Performance:\n")
    print(summary_df)

# Evaluate models for dogs
print("\nEvaluating Models for Dogs:")
X_dogs, y_dogs = prepare_data(df_dogs)
evaluate_models(X_dogs, y_dogs, classifiers, param_grids, retune_hyperparameters, pet_type=1)

# Evaluate models for cats
print("\nEvaluating Models for Cats:")
X_cats, y_cats = prepare_data(df_cats)
evaluate_models(X_cats, y_cats, classifiers, param_grids, retune_hyperparameters, pet_type=2)
