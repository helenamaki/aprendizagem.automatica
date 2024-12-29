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

# Set the target variable
y = df['AdoptionBinary']

# Perform PCA on the .contains columns
contains_columns = [col for col in X.columns if col.endswith('.contains')]
X_contains = X[contains_columns]
X_non_contains = X.drop(contains_columns, axis=1)

# Apply PCA to the .contains columns
pca = PCA(n_components=5)
X_contains_pca = pca.fit_transform(X_contains)

# Print the amount of variance explained by 5 components
print(f"Total variance explained by 5 components: {np.sum(pca.explained_variance_ratio_):.4f}")

# Combine PCA results with the non-.contains columns
X_final = np.concatenate([X_non_contains, X_contains_pca], axis=1)

# Define classifiers and parameter grids for tuning
classifiers = {
    'RandomForest': RandomForestClassifier(random_state=44, n_jobs=-1),  # Added n_jobs=-1 for parallelization
    'DecisionTree': DecisionTreeClassifier(random_state=44),  # n_jobs=-1 doesn't apply directly here
    'LogisticRegression': LogisticRegression(random_state=44, n_jobs=-1),  # Added n_jobs=-1 for parallelization
    'NaiveBayes': GaussianNB(),
    'KNeighbors': KNeighborsClassifier(n_jobs=-1),  # Added n_jobs=-1 for parallelization
    'SVC': SVC(C=1, gamma="auto"),  # No grid search yet, just using these parameters
    'Ensemble': VotingClassifier(estimators=[
        ('rf', RandomForestClassifier(random_state=44, n_jobs=-1)),  # Added n_jobs=-1 for parallelization
        ('dt', DecisionTreeClassifier(random_state=44)),
        ('knn', KNeighborsClassifier(n_jobs=-1))  # Added n_jobs=-1 for parallelization
    ], voting='hard')
}

# Define parameter grids for tuning
param_grids = {
    'RandomForest': {
        'n_estimators': [30, 60],  # Reduced options for faster tuning
        'max_features': ['sqrt'],  # Fixed the issue with 'auto'
        'max_depth': [10, 20, 30],  # Allow depth up to 30
        'min_samples_split': [2],
        'min_samples_leaf': [1],
        'bootstrap': [True]
    },
    'DecisionTree': {
        'max_depth': [10, 20, 30],  # Allow depth up to 30
        'min_samples_split': [2],
        'min_samples_leaf': [1],
        'criterion': ['gini']
    },
    'LogisticRegression': {
        'C': [0.1, 1],
        'solver': ['liblinear'],  # Use liblinear for smaller datasets
        'max_iter': [100]
    },
    'KNeighbors': {
        'n_neighbors': [3, 5],
        'weights': ['uniform'],
        'metric': ['euclidean']
    },
    'SVC': {
        'C': [1],  # Fixed C value
        'gamma': ['auto'],  # Auto gamma setting for SVC
        'kernel': ['rbf', 'linear', 'poly']  # Testing all three kernels
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

        # Set up GridSearchCV for the current model with parallelization
        grid_search = GridSearchCV(clf, param_grids[model_name], cv=5, scoring='accuracy', n_jobs=-1)  # n_jobs=-1 for parallelization
        grid_search.fit(X_final, y)

        # Get the best parameters and model
        best_params = grid_search.best_params_

        # Create the model with the best parameters
        clf.set_params(**best_params)
        print(f"Best hyperparameters for {model_name}: {best_params}")
    else:
        print(f"Skipping tuning for {model_name}, using default parameters.")
    
    # Perform cross-validation with parallelization
    print(f"\nEvaluating {model_name}...")
    cv_results = cross_val_score(clf, X_final, y, cv=5, scoring='accuracy', n_jobs=-1)  # n_jobs=-1 for parallelization
    print(f"Cross-Validation Accuracy Scores: {cv_results}")
    print(f"Mean Accuracy: {cv_results.mean():.4f}")
    print(f"Standard Deviation: {cv_results.std():.4f}")
    
    # Fit the model and predict
    clf.fit(X_final, y)
    y_pred = clf.predict(X_final)
    
    # Confusion Matrix
    cm = confusion_matrix(y, y_pred)
    print(f"Confusion Matrix for {model_name}:\n{cm}")
    
    # Classification Report
    report = classification_report(y, y_pred)
    print(f"Classification Report for {model_name}:\n{report}")

# Example of evaluating ensemble (VotingClassifier)
ensemble_clf = classifiers['Ensemble']
ensemble_clf.fit(X_final, y)
y_pred_ensemble = ensemble_clf.predict(X_final)

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
    cv_results = cross_val_score(clf, X_final, y, cv=5, scoring='accuracy', n_jobs=-1)  # n_jobs=-1 for parallelization
    results['Model'].append(model_name)
    results['Mean Accuracy'].append(cv_results.mean())
    results['Std Accuracy'].append(cv_results.std())

# Convert results to DataFrame for easy visualization
summary_df = pd.DataFrame(results)
summary_df = summary_df.sort_values(by='Mean Accuracy', ascending=False)
print("\nSummary of Model Performance:\n")
print(summary_df)
