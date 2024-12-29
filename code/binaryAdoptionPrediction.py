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
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import multiprocessing

# Load dataset
df = pd.read_csv('../data/Processed_PetFinder_dataset.csv')

# Create the new binary target (1 for adoption, 0 for not adopted)
df['AdoptionBinary'] = df['AdoptionSpeed'].apply(lambda x: 1 if x != 4 else 0)

# Prepare features (X) and target (y)
# Drop the target columns from the features
X = df.drop(['AdoptionSpeed', 'AdoptionBinary'], axis=1)

# Separate categorical and numerical features
categorical_vars = X.select_dtypes(include=['object']).columns.tolist()
numerical_vars = X.select_dtypes(exclude=['object']).columns.tolist()

# Apply one-hot encoding to categorical variables and keep the numerical ones as is
X_categorical = pd.get_dummies(X[categorical_vars], drop_first=True)
X_numerical = X[numerical_vars]

# Combine numerical and categorical features back together
X = pd.concat([X_numerical, X_categorical], axis=1)

# Standardize the data before applying PCA
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA to reduce dimensionality
pca = PCA(n_components=0.90)  # Keep 90% of the variance
X_pca = pca.fit_transform(X_scaled)
print(f"Original shape: {X_scaled.shape}")
print(f"Reduced shape after PCA: {X_pca.shape}")

# Set the target variable
y = df['AdoptionBinary']

# Function to save hyperparameters to JSON file
def save_hyperparameters(model_name, hyperparameters, filename='../data/binary/best_hyperparameters.json'):
    try:
        with open(filename, 'r') as f:
            all_params = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        all_params = {}
    
    all_params[model_name] = hyperparameters
    
    with open(filename, 'w') as f:
        json.dump(all_params, f, indent=4)

# Function to load hyperparameters from JSON file
def load_hyperparameters(model_name, filename='../data/binary/best_hyperparameters.json'):
    try:
        with open(filename, 'r') as f:
            all_params = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None
    
    return all_params.get(model_name, None)

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

# Hyperparameter tuning flag
retune_hyperparameters = True  # Change this flag to False to skip tuning

# Perform hyperparameter tuning (GridSearchCV) if necessary
for model_name, clf in classifiers.items():
    if model_name == "NaiveBayes" or model_name == "Ensemble":
        continue
    # Check if we need to retune
    if retune_hyperparameters:
        print(f"\nTuning hyperparameters for {model_name}...")

        # Check if hyperparameters are already saved
        best_params = load_hyperparameters(model_name)

        if best_params is None:
            print(f"No saved hyperparameters found for {model_name}, starting grid search...")

            # Set up GridSearchCV for the current model with parallelization
            grid_search = GridSearchCV(clf, param_grids[model_name], cv=5, scoring='accuracy', n_jobs=-1)  # n_jobs=-1 for parallelization
            grid_search.fit(X_pca, y)  # Use PCA-transformed data

            # Get the best parameters and model
            best_params = grid_search.best_params_

            # Save the best hyperparameters
            save_hyperparameters(model_name, best_params)
            print(f"Best hyperparameters for {model_name}: {best_params}")
        else:
            print(f"Using saved hyperparameters for {model_name}: {best_params}")
        
        # Create the model with the best parameters
        clf.set_params(**best_params)
    else:
        print(f"Skipping tuning for {model_name}, using default parameters.")
    
    # Perform cross-validation with parallelization
    print(f"\nEvaluating {model_name}...")
    cv_results = cross_val_score(clf, X_pca, y, cv=5, scoring='accuracy', n_jobs=-1)  # Use PCA-transformed data
    print(f"Cross-Validation Accuracy Scores: {cv_results}")
    print(f"Mean Accuracy: {cv_results.mean():.4f}")
    print(f"Standard Deviation: {cv_results.std():.4f}")
    
    # Fit the model and predict
    clf.fit(X_pca, y)  # Use PCA-transformed data
    y_pred = clf.predict(X_pca)  # Use PCA-transformed data
    
    # Confusion Matrix
    cm = confusion_matrix(y, y_pred)
    print(f"Confusion Matrix for {model_name}:\n{cm}")
    
    # Classification Report
    report = classification_report(y, y_pred)
    print(f"Classification Report for {model_name}:\n{report}")

# Example of evaluating ensemble (VotingClassifier)
ensemble_clf = classifiers['Ensemble']
ensemble_clf.fit(X_pca, y)  # Use PCA-transformed data
y_pred_ensemble = ensemble_clf.predict(X_pca)  # Use PCA-transformed data

# Confusion Matrix for Ensemble
cm_ensemble = confusion_matrix(y, y_pred_ensemble)
print(f"Confusion Matrix for Ensemble:\n{cm_ensemble}")

# Classification Report for Ensemble
report_ensemble = classification_report(y, y_pred_ensemble)
print(f"Classification Report for Ensemble:\n{report_ensemble}")

# Optionally: Save all results in a summary dataframe
results = {
    'Model': [],
    'Mean Accuracy': [],
    'Std Accuracy': []
}

# Cross-validation results and performance summary
for model_name, clf in classifiers.items():
    cv_results = cross_val_score(clf, X_pca, y, cv=5, scoring='accuracy', n_jobs=-1)  # Use PCA-transformed data
    results['Model'].append(model_name)
    results['Mean Accuracy'].append(cv_results.mean())
    results['Std Accuracy'].append(cv_results.std())

# Convert results to DataFrame for easy visualization
summary_df = pd.DataFrame(results)
summary_df = summary_df.sort_values(by='Mean Accuracy', ascending=False)
print("\nSummary of Model Performance:\n")
print(summary_df)
