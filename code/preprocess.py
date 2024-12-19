import pandas as pd
import os
from nltk.tokenize import TreebankWordTokenizer
import nltk

# Ensure NLTK is properly installed and setup
nltk.download('punkt')

# Set the current working directory to the directory of the script
current_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_dir)

# File paths
state_labels_file = 'state_labels.csv'
color_labels_file = 'color_labels.csv'
petfinder_file = 'PetFinder_dataset.csv'

# Read the datasets
state_labels = pd.read_csv(state_labels_file)
color_labels = pd.read_csv(color_labels_file)
petfinder_data = pd.read_csv(petfinder_file)

# Create mapping dictionaries
state_dict = dict(zip(state_labels['StateID'], state_labels['StateName']))
color_dict = dict(zip(color_labels['ColorID'], color_labels['ColorName']))

# Add "0" = "null" to color_dict
color_dict[0] = "null"

# Replace StateID with StateName
petfinder_data['State'] = petfinder_data['State'].replace(state_dict)

# Use TreebankWordTokenizer for tokenization
tokenizer = TreebankWordTokenizer()

# Tokenize the text in the Description column
petfinder_data['Description'] = petfinder_data['Description'].apply(
    lambda x: tokenizer.tokenize(x) if pd.notnull(x) else x
)

# Replace color columns with corresponding labels
color_columns = ['Color1', 'Color2', 'Color3']
for col in color_columns:
    petfinder_data[col] = petfinder_data[col].replace(color_dict)

# Save the processed dataset to a new file
output_file = 'Processed_PetFinder_dataset.csv'
petfinder_data.to_csv(output_file, index=False)

print(f"Processed dataset saved to {output_file}")
