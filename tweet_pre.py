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

def clean_text(text):
    if not isinstance(text, str):
        text = ''
    words = text.split()
    cleaned_words = [word.lower() for word in words if word.lower() not in nltk_stopwords]
    return ' '.join(cleaned_words)

def process_chunk(chunk):
    chunk['tweet_text_cleaned'] = chunk['tweet_text'].apply(clean_text)
    return chunk

def multiprocess_dataframe(df, num_partitions=4):
    df_split = np.array_split(df, num_partitions)
    with Pool(num_partitions) as pool:
        df_chunks = pool.map(process_chunk, df_split)
    return pd.concat(df_chunks).reset_index(drop=True)

def process_entire_dataset(file_path, chunk_size=50000):
    total_rows = 0
    rows_after_cleaning = 0
    chunk_count = 0

    # Iterate over chunks of the dataset without skipping any rows
    for chunk in pd.read_csv(file_path, usecols=['tweet_text'], chunksize=chunk_size):
        chunk_count += 1
        total_rows += len(chunk)
        print(f"Processing chunk {chunk_count}...")

        # Step 1: Print rows before cleaning
        print(f"Rows before cleaning in chunk {chunk_count}: {len(chunk)}")

        # Step 2: Clean the chunk (remove NaN or empty 'tweet_text')
        chunk_cleaned = chunk.dropna(subset=['tweet_text'], how='all')
        chunk_cleaned['tweet_text'] = chunk_cleaned['tweet_text'].astype(str)  # Convert to string to avoid errors
        chunk_cleaned = chunk_cleaned[(chunk_cleaned['tweet_text'].str.strip() != '')]
        
        rows_after_cleaning += len(chunk_cleaned)
        print(f"Rows after cleaning in chunk {chunk_count}: {len(chunk_cleaned)}")

        # Step 3: Remove rows with links in 'tweet_text'
        chunk_cleaned = chunk_cleaned[~chunk_cleaned['tweet_text'].str.contains(r'http[s]?://\S+', na=False)]

        # Step 4: Process the cleaned chunk using multiple processors
        chunk_processed = multiprocess_dataframe(chunk_cleaned, num_partitions=4)

    print(f"Total rows before cleaning: {total_rows}")
    print(f"Total rows after cleaning: {rows_after_cleaning}")

# Execute the function on the entire dataset
process_entire_dataset(
    "/scratch/general/vast/u1472278/tweets.csv",
    chunk_size=50000
)
