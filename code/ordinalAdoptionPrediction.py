import pandas as pd
import numpy as np
import json
from sklearn.model_selection import KFold, GridSearchCV
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
    'RandomForest': RandomForestClassifier(random_state=44, n_jobs=-1, max_depth=12, n_estimators=70), 
    'DecisionTree': DecisionTreeClassifier(random_state=44, max_depth=15), 
    'LogisticRegression': LogisticRegression(random_state=44, max_iter=100), 
    'NaiveBayes': GaussianNB(),  
    'KNeighbors': KNeighborsClassifier(n_jobs=-1, n_neighbors=7), 
    'SVC': SVC(),  
    'Ensemble': VotingClassifier(estimators=[
        ('rf', RandomForestClassifier(random_state=44, max_depth=15, n_estimators=70)),
        ('dt', DecisionTreeClassifier(random_state=44, max_depth=15)),
        ('knn', KNeighborsClassifier(n_jobs=-1, n_neighbors=7))
    ], voting='hard')
}

# Define parameter grids for tuning (with slightly increased values)
param_grids = {
    'RandomForest': {
        'n_estimators': [50, 70],  
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

# Set Pandas options to display the full width of the DataFrame
pd.set_option('display.width', None)  # No limit on the width
pd.set_option('display.max_columns', None)  # Show all columns
pd.set_option('display.expand_frame_repr', False)  # Do not truncate long DataFrame display in console

# Evaluate individual models first
for model_name, clf in classifiers.items():
    if model_name == 'Ensemble':
        continue
    clf.fit(X_final, y)
    y_pred = clf.predict(X_final)
    
    # Calculate MAE
    mae = mean_absolute_error(y, y_pred)
    
    # Calculate Accuracy
    accuracy = accuracy_score(y, y_pred)  # No multiplication by 100 (stays in decimal form)
    
    # Get confusion matrix and classification report
    cm = confusion_matrix(y, y_pred)
    cr = classification_report(y, y_pred)
    
    print(f"\nEvaluating {model_name}...")
    print(f"Confusion Matrix for {model_name}:\n{cm}")
    print(f"Classification Report for {model_name}:\n{cr}")
    
    # Perform KFold cross-validation and store results
    kf = KFold(n_splits=5, shuffle=True, random_state=44)
    mae_scores = []
    accuracy_scores = []
    for train_idx, test_idx in kf.split(X_final):
        X_train, X_test = X_final[train_idx], X_final[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        
        mae_scores.append(mean_absolute_error(y_test, y_pred))
        accuracy_scores.append(accuracy_score(y_test, y_pred))
    
    print(f"KFold MAE scores for {model_name}: {mae_scores}")
    print(f"KFold Accuracy scores for {model_name}: {accuracy_scores}")
    
    # Calculate standard deviation of MAE and accuracy
    std_mae = np.std(mae_scores)
    std_accuracy = np.std(accuracy_scores)
    
    # Store results in the summary
    summary.append({
        'Model': model_name,
        'Mean Absolute Error (MAE)': mae,
        'Accuracy': accuracy,
        'Std MAE': std_mae,
        'Std Accuracy': std_accuracy
    })

# Now add the ensemble model to the summary
clf_ensemble = classifiers['Ensemble']
clf_ensemble.fit(X_final, y)
y_pred_ensemble = clf_ensemble.predict(X_final)
ensemble_mae = mean_absolute_error(y, y_pred_ensemble)
ensemble_accuracy = accuracy_score(y, y_pred_ensemble)
ensemble_cm = confusion_matrix(y, y_pred_ensemble)
ensemble_cr = classification_report(y, y_pred_ensemble)

print(f"\nEvaluating Ensemble (VotingClassifier)...")
print(f"Confusion Matrix for Ensemble (VotingClassifier):\n{ensemble_cm}")
print(f"Classification Report for Ensemble (VotingClassifier):\n{ensemble_cr}")

# Perform KFold for the ensemble model
kf_ensemble = KFold(n_splits=5, shuffle=True, random_state=44)
ensemble_mae_scores = []
ensemble_accuracy_scores = []
for train_idx, test_idx in kf_ensemble.split(X_final):
    X_train, X_test = X_final[train_idx], X_final[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
    clf_ensemble.fit(X_train, y_train)
    y_pred = clf_ensemble.predict(X_test)
    
    ensemble_mae_scores.append(mean_absolute_error(y_test, y_pred))
    ensemble_accuracy_scores.append(accuracy_score(y_test, y_pred))

print(f"KFold MAE scores for Ensemble (VotingClassifier): {ensemble_mae_scores}")
print(f"KFold Accuracy scores for Ensemble (VotingClassifier): {ensemble_accuracy_scores}")

# Calculate standard deviation for the ensemble
ensemble_std_mae = np.std(ensemble_mae_scores)
ensemble_std_accuracy = np.std(ensemble_accuracy_scores)

# Store results for the ensemble model
summary.append({
    'Model': 'Ensemble (VotingClassifier)',
    'Mean Absolute Error (MAE)': ensemble_mae,
    'Accuracy': ensemble_accuracy,
    'Std MAE': ensemble_std_mae,
    'Std Accuracy': ensemble_std_accuracy
})

# Print out a summary table, sorted by MAE
summary_df = pd.DataFrame(summary)

# Sort by Mean Absolute Error (lower is better)
summary_df = summary_df.sort_values(by='Mean Absolute Error (MAE)', ascending=True)

# Print the full summary table
print("\nSummary of Model Performance:")
print(summary_df)
