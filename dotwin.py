import pandas as pd
import re
from nltk.corpus import stopwords
import nltk
from multiprocessing import Pool, cpu_count
import numpy as np

# Download NLTK stopwords if you haven't already
nltk.download('stopwords')

# Load NLTK English stopwords
nltk_stopwords = set(stopwords.words('english'))

# Load the dataset
df = pd.read_csv("datasets/dotwin_comments.csv")

# Step 1: Remove rows where 'content' is NaN or empty
df_cleaned = df.dropna(subset=['content'], how='all')  # Drop rows if 'content' is NaN
df_cleaned = df_cleaned[(df_cleaned['content'].str.strip() != '')]  # Keep rows with non-empty 'content'

# Step 2: Remove rows with links in 'content' (detecting links using regex)
df_cleaned = df_cleaned[~df_cleaned['content'].str.contains(r'http[s]?://\S+', na=False)]

# Reset the index of the dataframe
df_cleaned = df_cleaned.reset_index(drop=True)

# Dictionaries for category matching (as provided)
fusion = ["brother", "sister", "family", "motherland", "our blood", "fatherland", "sons", "daughters", "kin", "my people", "my race", "our people", "European race", "ancestry", "ancestor", "descendant", "fellow", "brethren", "comrades"]
violence = ["kill", "hang", "bomb", "shoot", "slaughter", "execute", "execution", "punish", "death penalty", "massacre", "destroy", "must attack", "must fight", "revenge", "retribution", "eradicate", "starve", "die", "torture", "behead", "burn", "bring death to", "give them hell", "weapon", "firearm", "assassinate", "gun", "rifle", "knife", "grenade", "brutal steps", "molotov", "jihaad", "jihad", "set fire", "revolution", "forcible overthrow", "flamethrowers", "M1-16", "ammonium nitrate"]
# Additional dictionaries...

# Add all dictionaries to a list for easier processing
dictionaries = {
    'fusion': fusion,
    'violence': violence,
    # Add other dictionaries as needed...
}

# Pre-compile regex patterns for each dictionary to improve performance
compiled_patterns = {category: [re.compile(rf'\b{re.escape(word)}\b') for word in words_list]
                     for category, words_list in dictionaries.items()}

# Function to clean text: remove stop words and lower the text
def clean_text(text):
    # Convert non-string values to an empty string
    if not isinstance(text, str):
        text = ''
    words = text.split()
    cleaned_words = [word.lower() for word in words if word.lower() not in nltk_stopwords]
    return ' '.join(cleaned_words)

# Function to count occurrences from each dictionary in the text
def count_categories(text):
    category_counts = {category: 0 for category in dictionaries}  # Initialize counts
    for category, patterns in compiled_patterns.items():
        for pattern in patterns:
            if pattern.search(text):
                category_counts[category] += 1
    return category_counts

# Function to count the number of tokens (words) in cleaned text
def count_tokens(text):
    return len(text.split())

# Function to process a chunk of the DataFrame
def process_chunk(chunk):
    chunk['content_cleaned'] = chunk['content'].apply(clean_text)
    chunk['content_category_counts'] = chunk['content_cleaned'].apply(count_categories)
    chunk['content_token_count'] = chunk['content_cleaned'].apply(count_tokens)
    return chunk

# Function to split the DataFrame into chunks and process them in parallel
def multiprocess_dataframe(df, num_partitions=None):
    if num_partitions is None:
        num_partitions = cpu_count()  # Use the number of CPUs available
    
    # Split the DataFrame into smaller chunks for parallel processing
    df_split = np.array_split(df, num_partitions)
    
    # Use a pool of workers to process the chunks
    with Pool(num_partitions) as pool:
        df_chunks = pool.map(process_chunk, df_split)
    
    # Combine all the processed chunks back into a single DataFrame
    return pd.concat(df_chunks).reset_index(drop=True)

# Function to handle large dataset and process in chunks of 100k rows
def process_in_batches(df, batch_size=100000):
    total_rows = df.shape[0]
    total_batches = (total_rows // batch_size) + (1 if total_rows % batch_size != 0 else 0)
    final_results = []  # This will accumulate all results

    for i in range(total_batches):
        print(f"Processing batch {i+1}/{total_batches}...")
        start = i * batch_size
        end = min((i + 1) * batch_size, total_rows)
        df_batch = df.iloc[start:end]
        # Process using multiple processors
        df_batch_cleaned = multiprocess_dataframe(df_batch)
        # Append results from this batch to final results
        final_results.append(df_batch_cleaned)
        print(f"Batch {i+1} processed with {len(df_batch_cleaned)} rows.")

    # Combine all final results into a single DataFrame
    return pd.concat(final_results).reset_index(drop=True)

# Function to save the final token count and dictionary counts to a .txt file
def save_final_summary(df, output_file='final_summary.txt'):
    # Total number of tokens
    total_tokens = df['content_token_count'].sum()

    # Combined dictionary counts
    combined_category_counts_total = df['content_category_counts'].apply(pd.Series).sum()

    with open(output_file, 'w') as f:
        f.write(f"Total number of tokens from content: {total_tokens}\n")
        f.write("Total combined category counts for content column:\n")
        f.write(f"{combined_category_counts_total}\n")

    print(f"Final summary saved to {output_file}.")

# Process the dataset in batches of 100k rows
processed_df = process_in_batches(df_cleaned, batch_size=100000)

# Save final token count and combined dictionary counts to a text file
save_final_summary(processed_df, output_file='final_summary.txt')
