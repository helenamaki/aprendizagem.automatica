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
import os

# Load dataset
df = pd.read_csv('../data/Processed_PetFinder_dataset.csv')

# Filter for cats (Type == 1) and dogs (Type == 2)
df_cats = df[df['Type'] == 1]
df_dogs = df[df['Type'] == 2]

# Define a function to run the model for a given subset and save results
def run_model_on_subset(df_subset, output_folder):
    # Create the new binary target (1 for adoption, 0 for not adopted)
    df_subset['AdoptionBinary'] = df_subset['AdoptionSpeed'].apply(lambda x: 1 if x != 4 else 0)

    # Prepare features (X) and target (y)
    X = df_subset.drop(['AdoptionSpeed', 'AdoptionBinary', 'Type'], axis=1)
    y = df_subset['AdoptionBinary']

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

    # Define classifiers and parameter grids for tuning
    classifiers = {
        'RandomForest': RandomForestClassifier(random_state=44, n_jobs=-1, max_depth=15, n_estimators=50),  # Increase depth & estimators
        'DecisionTree': DecisionTreeClassifier(random_state=44, max_depth=15),  # Increase depth
        'LogisticRegression': LogisticRegression(random_state=44, max_iter=100),  # Keep it simple
        'NaiveBayes': GaussianNB(),  # NaiveBayes is already fast
        'KNeighbors': KNeighborsClassifier(n_jobs=-1, n_neighbors=7),  # Increase neighbors
        'SVC': SVC(),  # Placeholder for SVC
        'Ensemble': VotingClassifier(estimators=[
            ('rf', RandomForestClassifier(random_state=44, max_depth=15, n_estimators=100)),
            ('dt', DecisionTreeClassifier(random_state=44, max_depth=15)),
            ('knn', KNeighborsClassifier(n_jobs=-1, n_neighbors=7))
        ], voting='hard')
    }

    # Define parameter grids for tuning (with slightly increased values)
    param_grids = {
        'RandomForest': {
            'n_estimators': [50, 100],  # Increased number of estimators for a better model
            'max_features': ['sqrt'],  # Keep the number of features low
            'max_depth': [10, 15],  # Increased depth slightly
            'min_samples_split': [2],  # Keep the split simple
            'min_samples_leaf': [1],  # Keep the leaf size small
            'bootstrap': [True]
        },
        'DecisionTree': {
            'max_depth': [10, 12],  # Increased depth slightly
            'min_samples_split': [2],
            'min_samples_leaf': [1],
            'criterion': ['gini', 'entropy']
        },
        'LogisticRegression': {
            'C': [0.1, 1],  # Limit search space for regularization strength
            'solver': ['liblinear'],  # Choose faster solver
            'max_iter': [100]  # Limit iterations for faster convergence
        },
        'KNeighbors': {
            'n_neighbors': [5, 7],  # Increased neighbors slightly
            'weights': ['uniform'],
            'metric': ['euclidean']  # Keep distance metric simple
        },
        'SVC': {
            'C': [0.1, 1, 10],  # Tune C with reasonable values
            'gamma': ['auto', 'scale', 0.1],  # Tune gamma with reasonable values
            'kernel': ['rbf']  # Stick to rbf kernel for efficiency
        }
    }

    # File path for saving/loading hyperparameters
    hyperparameters_file = os.path.join(output_folder, 'hyperparameters.json')

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
        # Check if we need to retune
        if model_name in saved_hyperparameters and not retune_hyperparameters:
            print(f"Using saved hyperparameters for {model_name}.")
            clf.set_params(**saved_hyperparameters[model_name])
        else:
            print(f"\nTuning hyperparameters for {model_name}...")

            # Set up GridSearchCV for the current model with parallelization
            grid_search = GridSearchCV(clf, param_grids[model_name], cv=5, scoring='accuracy', n_jobs=-1)  # n_jobs=-1 for parallelization
            grid_search.fit(X_final, y)

            # Get the best parameters and model
            best_params = grid_search.best_params_

            # Save the best parameters for later use
            saved_hyperparameters[model_name] = best_params

            # Create the model with the best parameters
            clf.set_params(**best_params)
            print(f"Best hyperparameters for {model_name}: {best_params}")

    # Save hyperparameters to JSON file after tuning
    with open(hyperparameters_file, 'w') as f:
        json.dump(saved_hyperparameters, f, indent=4)
        print(f"Saved hyperparameters to {hyperparameters_file}")

    # Perform cross-validation with parallelization and print results
    for model_name, clf in classifiers.items():
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

    # Save summary to file
    summary_df.to_csv(os.path.join(output_folder, 'model_performance_summary.csv'), index=False)
    print(f"Saved model performance summary to {output_folder}/model_performance_summary.csv")

# Create output folders for cats and dogs if they don't exist
os.makedirs('binaryCats', exist_ok=True)
os.makedirs('binaryDogs', exist_ok=True)

# Run the model for cats and dogs separately
run_model_on_subset(df_cats, 'binaryCats')
run_model_on_subset(df_dogs, 'binaryDogs')
