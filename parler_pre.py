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
    chunk['body_cleaned'] = chunk['body'].apply(clean_text)
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

    # Iterate over all chunks in the dataset
    for chunk in pd.read_csv(file_path, usecols=['body'], chunksize=chunk_size):
        chunk_count += 1
        total_rows += len(chunk)
        print(f"Processing chunk {chunk_count}...")

        # Print rows before cleaning
        print(f"Rows before cleaning in chunk {chunk_count}: {len(chunk)}")

        # Clean the chunk by removing NaN or empty 'body' values
        chunk_cleaned = chunk.dropna(subset=['body'], how='all')
        chunk_cleaned['body'] = chunk_cleaned['body'].astype(str)  # Convert to string to avoid errors
        chunk_cleaned = chunk_cleaned[(chunk_cleaned['body'].str.strip() != '')]

        rows_after_cleaning += len(chunk_cleaned)
        print(f"Rows after cleaning in chunk {chunk_count}: {len(chunk_cleaned)}")

        # Remove rows with links in 'body'
        chunk_cleaned = chunk_cleaned[~chunk_cleaned['body'].str.contains(r'http[s]?://\S+', na=False)]

        # Process the cleaned chunk using multiple processors
        chunk_processed = multiprocess_dataframe(chunk_cleaned, num_partitions=4)

    # Print total row counts before and after cleaning
    print(f"Total rows before cleaning: {total_rows}")
    print(f"Total rows after cleaning: {rows_after_cleaning}")

# Execute the function on the entire dataset
process_entire_dataset(
    "/scratch/general/vast/u1472278/parler_posts_comments.csv",
    chunk_size=50000
)
