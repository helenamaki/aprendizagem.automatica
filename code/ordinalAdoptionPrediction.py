import pandas as pd
import numpy as np
import json
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, mean_absolute_error
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.decomposition import PCA
import os

# Load dataset
df = pd.read_csv('../data/Processed_PetFinder_dataset.csv')

# Prepare features (X) and target (y)
X = df.drop(['AdoptionSpeed'], axis=1)  # No need for AdoptionBinary anymore
y = df['AdoptionSpeed']

# Separate categorical and numerical features
categorical_vars = X.select_dtypes(include=['object']).columns.tolist()
numerical_vars = X.select_dtypes(exclude=['object']).columns.tolist()

# Apply one-hot encoding to categorical variables and keep the numerical ones as is
X_categorical = pd.get_dummies(X[categorical_vars], drop_first=True)
X_numerical = X[numerical_vars]

# Combine numerical and categorical features back together
X = pd.concat([X_numerical, X_categorical], axis=1)

# Perform PCA on the .contains columns
contains_columns = [col for col in X.columns if col.endswith('.contains')]
X_contains = X[contains_columns]
X_non_contains = X.drop(contains_columns, axis=1)

# Apply PCA with fewer components for faster computation
pca = PCA(n_components=5)  # Use 5 components to speed up PCA
X_contains_pca = pca.fit_transform(X_contains)

# Print the amount of variance explained by the 5 components
print(f"Total variance explained by 5 components: {np.sum(pca.explained_variance_ratio_):.4f}")

# Combine PCA results with the non-.contains columns
X_final = np.concatenate([X_non_contains, X_contains_pca], axis=1)

# Define classifiers
classifiers = {
    'RandomForest': RandomForestClassifier(random_state=44, n_jobs=-1, max_depth=15, n_estimators=50), 
    'DecisionTree': DecisionTreeClassifier(random_state=44, max_depth=15), 
    'LogisticRegression': LogisticRegression(random_state=44, max_iter=100), 
    'NaiveBayes': GaussianNB(),  
    'KNeighbors': KNeighborsClassifier(n_jobs=-1, n_neighbors=7), 
    'SVC': SVC(),  
    'Ensemble': VotingClassifier(estimators=[
        ('rf', RandomForestClassifier(random_state=44, max_depth=15, n_estimators=100)),
        ('dt', DecisionTreeClassifier(random_state=44, max_depth=15)),
        ('knn', KNeighborsClassifier(n_jobs=-1, n_neighbors=7))
    ], voting='hard')
}

# Define parameter grids for tuning (with slightly increased values)
param_grids = {
    'RandomForest': {
        'n_estimators': [50, 80],  
        'max_features': ['sqrt'],  
        'max_depth': [10, 12],  
        'min_samples_split': [2],  
        'min_samples_leaf': [1],  
        'bootstrap': [True]
    },
    'DecisionTree': {
        'max_depth': [10, 12],  
        'min_samples_split': [2],
        'min_samples_leaf': [1],
        'criterion': ['gini', 'entropy']
    },
    'LogisticRegression': {
        'C': [0.1, 1],  
        'solver': ['liblinear'],  
        'max_iter': [100]  
    },
    'KNeighbors': {
        'n_neighbors': [5, 7],  
        'weights': ['uniform'],
        'metric': ['euclidean']  
    },
    'SVC': {
        'C': [0.1, 1, 10],  
        'gamma': ['auto'],  
        'kernel': ['rbf']  
    }
}

# File path for saving/loading hyperparameters
hyperparameters_file = '../data/ordinal/hyperparameters.json'

# Load hyperparameters from JSON if they exist
if os.path.exists(hyperparameters_file):
    with open(hyperparameters_file, 'r') as f:
        saved_hyperparameters = json.load(f)
    print("Loaded saved hyperparameters from file.")
else:
    saved_hyperparameters = {}

# Hyperparameter tuning flag
retune_hyperparameters = True  # Change this flag to False to skip tuning

# Perform hyperparameter tuning (GridSearchCV) if necessary
for model_name, clf in classifiers.items():
    if model_name == "NaiveBayes" or model_name == "Ensemble":
        continue
    if model_name in saved_hyperparameters and not retune_hyperparameters:
        print(f"Using saved hyperparameters for {model_name}.")
        clf.set_params(**saved_hyperparameters[model_name])
    else:
        print(f"\nTuning hyperparameters for {model_name}...")
        grid_search = GridSearchCV(clf, param_grids[model_name], cv=5, scoring='accuracy', n_jobs=-1)  
        grid_search.fit(X_final, y)
        best_params = grid_search.best_params_
        saved_hyperparameters[model_name] = best_params
        clf.set_params(**best_params)
        print(f"Best hyperparameters for {model_name}: {best_params}")

# Save hyperparameters to JSON file after tuning
with open(hyperparameters_file, 'w') as f:
    json.dump(saved_hyperparameters, f, indent=4)
    print(f"Saved hyperparameters to {hyperparameters_file}")

# Collect summary of performance metrics (Accuracy and MAE)
summary = []

# Evaluate individual models first
for model_name, clf in classifiers.items():
    clf.fit(X_final, y)
    y_pred = clf.predict(X_final)
    
    # Calculate MAE
    mae = mean_absolute_error(y, y_pred)
    
    # Calculate Accuracy
    accuracy = accuracy_score(y, y_pred)  # No multiplication by 100 (stays in decimal form)
    
    print(f"MAE for {model_name}: {mae:.4f}, Accuracy: {accuracy:.4f}")
    summary.append({
        'Model': model_name,
        'Mean Absolute Error (MAE)': mae,
        'Accuracy': accuracy
    })

# Now add the ensemble model to the summary
ensemble_mae = mean_absolute_error(y, y_pred_ensemble)
ensemble_accuracy = accuracy_score(y, y_pred_ensemble)  # No multiplication by 100 (stays in decimal form)
summary.append({
    'Model': 'Ensemble (VotingClassifier)',
    'Mean Absolute Error (MAE)': ensemble_mae,
    'Accuracy': ensemble_accuracy
})

# Print out a summary table, sorted by MAE
summary_df = pd.DataFrame(summary)
summary_df = summary_df.sort_values(by='Mean Absolute Error (MAE)', ascending=True)  # Lower MAE is better
print("\nSummary of Model Performance (Ranked by MAE):")
print(summary_df)
