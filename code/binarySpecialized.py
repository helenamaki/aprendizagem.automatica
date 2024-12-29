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

def run_model_on_subset(df_subset, hyperparameters_file, animal_type):
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

    # Load hyperparameters if they exist
    if os.path.exists(hyperparameters_file):
        with open(hyperparameters_file, 'r') as f:
            saved_hyperparameters = json.load(f)
        print(f"Loaded saved hyperparameters from {hyperparameters_file}.")
    else:
        saved_hyperparameters = {}

    # Hyperparameter tuning flag
    retune_hyperparameters = False

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

            # Define parameter grids for tuning (for simplicity, some values are pre-defined)
            param_grids = {
                'RandomForest': {
                    'n_estimators': [50, 100],
                    'max_features': ['sqrt'],
                    'max_depth': [10, 15],
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
                    'gamma': ['auto', 'scale', 0.1],
                    'kernel': ['rbf']
                }
            }

            # Set up GridSearchCV for the current model with parallelization
            grid_search = GridSearchCV(clf, param_grids[model_name], cv=5, scoring='accuracy', n_jobs=-1)
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
    results = {'Model': [], 'Mean Accuracy': [], 'Std Accuracy': []}

    for model_name, clf in classifiers.items():
        print(f"\nEvaluating {model_name}...")
        cv_results = cross_val_score(clf, X_final, y, cv=5, scoring='accuracy', n_jobs=-1)
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

        # Save results for the summary dataframe
        results['Model'].append(model_name)
        results['Mean Accuracy'].append(cv_results.mean())
        results['Std Accuracy'].append(cv_results.std())

    # Convert results to DataFrame for easy visualization
    summary_df = pd.DataFrame(results)
    summary_df = summary_df.sort_values(by='Mean Accuracy', ascending=False)
    print("\nSummary of Model Performance:")
    print(summary_df)


# Main function for running both Dog and Cat models
def main():
    # Load the original dataset
    df = pd.read_csv('../data/Processed_PetFinder_dataset.csv')

    # Filter subsets for Dogs (Type == 1) and Cats (Type == 2)
    df_dogs = df[df['Type'] == 1]
    df_cats = df[df['Type'] == 2]

    # Run model for Dogs
    print("\nRunning model for Dogs (Type = 1)...")
    dog_hyperparameters_file = '../data/binary/DogHyperparameters.json'
    run_model_on_subset(df_dogs, dog_hyperparameters_file, 'dog')

    # Run model for Cats
    print("\nRunning model for Cats (Type = 2)...")
    cat_hyperparameters_file = '../data/binary/CatHyperparameters.json'
    run_model_on_subset(df_cats, cat_hyperparameters_file, 'cat')


# Call the main function
if __name__ == "__main__":
    main()
