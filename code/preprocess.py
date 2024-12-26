import pandas as pd
import os
from nltk.tokenize import TreebankWordTokenizer
import nltk
import string
from nltk.util import bigrams
from collections import defaultdict

# Ensure NLTK is properly installed and setup
nltk.download('punkt')

# Set the current working directory to the directory of the script
current_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_dir)

# File paths
state_labels_file = '../data/state_labels.csv'
color_labels_file = '../data/color_labels.csv'
petfinder_file = '../data/PetFinder_dataset.csv'

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

# Replace "Name" with a binary indicator (0 or 1) depending on if there is a name
petfinder_data['Name'] = petfinder_data['Name'].apply(lambda x: 1 if pd.notnull(x) and x.strip() != "" else 0)

# Replace RescuerID with the number of pets associated with that RescuerID
rescuer_pet_count = petfinder_data['RescuerID'].value_counts()
petfinder_data['Rescuer.NumRescues'] = petfinder_data['RescuerID'].map(rescuer_pet_count)

# Drop the old RescuerID column
petfinder_data.drop(columns=['RescuerID'], inplace=True)

# Use TreebankWordTokenizer for tokenization
# Convert text to lowercase before tokenization and remove punctuation
tokenizer = TreebankWordTokenizer()

def remove_punctuation(text):
    """Remove punctuation from the text."""
    return ''.join([char for char in text if char not in string.punctuation])

# Add a new column with the length of the description (number of tokens)
petfinder_data['Description.Length'] = petfinder_data['Description'].apply(lambda x: len(x) if isinstance(x, list) else 0)

# Tokenize the text in the Description column after removing punctuation
petfinder_data['Description'] = petfinder_data['Description'].apply(
    lambda x: tokenizer.tokenize(remove_punctuation(x.lower())) if pd.notnull(x) else x
)

# Drop the PetID column
petfinder_data.drop(columns=['PetID'], inplace=True)

# Replace color columns with corresponding labels
color_columns = ['Color1', 'Color2', 'Color3']
for col in color_columns:
    petfinder_data[col] = petfinder_data[col].replace(color_dict)

# Identify the set of all words that appear in more than 250 unique listings
def get_unique_listing_counts(descriptions):
    word_in_listings = defaultdict(set)
    bigram_in_listings = defaultdict(set)
    
    for idx, desc in enumerate(descriptions):
        if isinstance(desc, list):
            for word in set(desc):
                word_in_listings[word].add(idx)
            # Get bigrams and count their occurrences
            for bigram in bigrams(desc):
                bigram_in_listings[bigram].add(idx)
    
    word_counts = {word: len(indices) for word, indices in word_in_listings.items()}
    bigram_counts = {bigram: len(indices) for bigram, indices in bigram_in_listings.items()}
    
    return word_counts, bigram_counts

# Get word and bigram counts
word_counts, bigram_counts = get_unique_listing_counts(petfinder_data['Description'])

#Only include words / bigrams appearing in at least 250 listings
frequent_words = {word for word, count in word_counts.items() if count > 250}
frequent_bigrams = {bigram for bigram, count in bigram_counts.items() if count > 250}

# Add a column for each word indicating its presence in the description
for word in frequent_words:
    petfinder_data[word + '.contains'] = petfinder_data['Description'].apply(
        lambda x: 1 if isinstance(x, list) and word in x else 0
    )

# Add a column for each bigram indicating its presence in the description
for bigram in frequent_bigrams:
    bigram_name = ' '.join(bigram)  # Join the two words in the bigram to make a single string
    petfinder_data[bigram_name + '.contains'] = petfinder_data['Description'].apply(
        lambda x: 1 if isinstance(x, list) and bigram[0] in x and bigram[1] in x else 0
    )

# Remove the Description column
petfinder_data.drop(columns=['Description'], inplace=True)

# Reorder the DataFrame columns to have word and bigram columns first
word_columns = [col for col in petfinder_data.columns if col.endswith('.contains')]
all_columns = [col for col in petfinder_data.columns if not col.endswith('.contains')] + word_columns
petfinder_data = petfinder_data[all_columns]

# Save the processed dataset to a new file
output_file = '../data/Processed_PetFinder_dataset.csv'
petfinder_data.to_csv(output_file, index=False)

print(f"Processed dataset saved to {output_file}")

# Create a text file listing the frequent words and their counts
word_count_file = '../data/frequent_words_counts.txt'
with open(word_count_file, 'w') as f:
    for word in frequent_words:
        f.write(f"{word}: {word_counts[word]}\n")

print(f"Frequent words and their counts saved to {word_count_file}")

# Create a text file listing the frequent bigrams and their counts
bigram_count_file = '../data/frequent_bigrams_counts.txt'
with open(bigram_count_file, 'w') as f:
    for bigram in frequent_bigrams:
        f.write(f"{' '.join(bigram)}: {bigram_counts[bigram]}\n")

print(f"Frequent bigrams and their counts saved to {bigram_count_file}")
