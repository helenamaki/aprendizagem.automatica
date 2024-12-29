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
  
