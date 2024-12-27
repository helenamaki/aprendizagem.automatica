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
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.pipeline import Pipeline

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

# Function to save hyperparameters and configurations to JSON files
def save_to_json(data, filename):
    try:
        with open(filename, 'r') as f:
            existing_data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        existing_data = {}
    
    existing_data.update(data)
    
    with open(filename, 'w') as f:
        json.dump(existing_data, f, indent=4)

# Function to load hyperparameters and configurations from JSON files
def load_from_json(filename):
    try:
        with open(filename, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None

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

# Function for feature selection using SelectKBest
def feature_selection(X, y, k=10):
    selector = SelectKBest(f_classif, k=k)
    X_new = selector.fit_transform(X, y)
    selected_features = X.columns[selector.get_support()]
    return X_new, selected_features

# Hyperparameter tuning flag
retune_hyperparameters = True  # Change this flag to False to skip tuning

# Scaling and balancing flags
scale_data = True  # Use scaling or not
balance_classes = True  # Use class balancing (SMOTE) or not

# Load previous configurations if available
scaling_config = load_from_json('../data/binary/scaling_config.json')
feature_selection_config = load_from_json('../data/binary/feature_selection_config.json')
class_balancing_config = load_from_json('../data/binary/class_balancing_config.json')
model_params_config = load_from_json('../data/binary/best_hyperparameters.json')

# Feature Selection: Apply SelectKBest based on saved config or tune
if feature_selection_config is None:
    print("Performing feature selection...")
    X, selected_features = feature_selection(X, y, k=10)
    save_to_json({'selected_features': list(selected_features)}, '../data/binary/feature_selection_config.json')
else:
    selected_features = feature_selection_config.get('selected_features', X.columns.tolist())
    print(f"Using saved feature selection: {selected_features}")

# Scaling: Apply StandardScaler based on saved config or tune
if scale_data:
    if scaling_config is None:
        print("Scaling features using StandardScaler...")
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        save_to_json({'scaling_used': True}, '../data/binary/scaling_config.json')
    else:
        print("Using saved scaling configuration...")
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
else:
    X_scaled = X  # No scaling if scale_data is False

# Class Balancing: Apply SMOTE based on saved config or tune
if balance_classes:
    if class_balancing_config is None:
        print("Balancing classes using SMOTE...")
        smote = SMOTE(random_state=44)
        X_balanced, y_balanced = smote.fit_resample(X_scaled, y)
        save_to_json({'class_balancing_used': True}, '../data/binary/class_balancing_config.json')
    else:
        print("Using saved class balancing configuration...")
        X_balanced, y_balanced = X_scaled, y
else:
    X_balanced, y_balanced = X_scaled, y

# Perform hyperparameter tuning (GridSearchCV) if necessary for each model
for model_name, clf in classifiers.items():
    if model_name == "NaiveBayes" or model_name == "Ensemble":
        continue
    print(f"\nTuning hyperparameters for {model_name}...")

    # Load previously saved hyperparameters if they exist
    best_params = model_params_config.get(model_name) if model_params_config else None

    if best_params is None:
        print(f"No saved hyperparameters found for {model_name}, starting grid search...")
        grid_search = GridSearchCV(clf, param_grids[model_name], cv=5, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X_balanced, y_balanced)
        best_params = grid_search.best_params_
        save_to_json({model_name: best_params}, '../data/binary/best_hyperparameters.json')
        print(f"Best hyperparameters for {model_name}: {best_params}")
    else:
        print(f"Using saved hyperparameters for {model_name}: {best_params}")

    clf.set_params(**best_params)

    # Evaluate model with cross-validation
    print(f"\nEvaluating {model_name}...")
    cv_results = cross_val_score(clf, X_balanced, y_balanced, cv=5, scoring='accuracy', n_jobs=-1)
    print(f"Cross-Validation Accuracy Scores: {cv_results}")
    print(f"Mean Accuracy: {cv_results.mean():.4f}")
    print(f"Standard Deviation: {cv_results.std():.4f}")

    # Fit the model and predict
    clf.fit(X_balanced, y_balanced)
    y_pred = clf.predict(X_balanced)

    # Confusion Matrix and Classification Report
    cm = confusion_matrix(y_balanced, y_pred)
    print(f"Confusion Matrix for {model_name}:\n{cm}")
    report = classification_report(y_balanced, y_pred)
    print(f"Classification Report for {model_name}:\n{report}")

# Example of evaluating ensemble (VotingClassifier)
ensemble_clf = classifiers['Ensemble']
ensemble_clf.fit(X_balanced, y_balanced)
y_pred_ensemble = ensemble_clf.predict(X_balanced)

# Confusion Matrix for Ensemble
cm_ensemble = confusion_matrix(y_balanced, y_pred_ensemble)
print(f"Confusion Matrix for Ensemble:\n{cm_ensemble}")

# Classification Report for Ensemble
report_ensemble = classification_report(y_balanced, y_pred_ensemble)
print(f"Classification Report for Ensemble:\n{report_ensemble}")

# Optionally: Save all results in a summary dataframe
results = {
    'Model': [],
    'Mean Accuracy': [],
    'Std Accuracy': []
}

# Cross-validation results and performance summary
for model_name, clf in classifiers.items():
    if model_name == "Ensemble":
        continue  # Skip ensemble model here
    cv_results = cross_val_score(clf, X_balanced, y_balanced, cv=5, scoring='accuracy', n_jobs=-1)
    results['Model'].append(model_name)
    results['Mean Accuracy'].append(cv_results.mean())
    results['Std Accuracy'].append(cv_results.std())

# Add ensemble model to the results
results['Model'].append("Ensemble")
results['Mean Accuracy'].append(np.mean(cross_val_score(ensemble_clf, X_balanced, y_balanced, cv=5)))
results['Std Accuracy'].append(np.std(cross_val_score(ensemble_clf, X_balanced, y_balanced, cv=5)))

# Convert results to DataFrame for easy visualization
summary_df = pd.DataFrame(results)
summary_df = summary_df.sort_values(by='Mean Accuracy', ascending=False)

# Print summary of model performance
print("\nSummary of Model Performance:\n")
print(summary_df)
