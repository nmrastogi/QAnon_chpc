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

# Define start and end dates for filtering
start_date = "2020-09-28"
end_date = "2020-12-05"

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

    for chunk in pd.read_csv(file_path, usecols=['body', 'createdAtformatted'], chunksize=chunk_size):
        chunk_count += 1
        total_rows += len(chunk)
        print(f"\nProcessing chunk {chunk_count}...")
        print(f"Rows before filtering: {len(chunk)}")

        # Filter by date range (lexicographic works if format is YYYY-MM-DD)
        chunk = chunk[(chunk['createdAtformatted'] >= start_date) & (chunk['createdAtformatted'] <= end_date)]
        print(f"Rows after date filter: {len(chunk)}")

        # Drop NaN or empty body
        chunk_cleaned = chunk.dropna(subset=['body'], how='all')
        chunk_cleaned['body'] = chunk_cleaned['body'].astype(str)
        chunk_cleaned = chunk_cleaned[chunk_cleaned['body'].str.strip() != '']

        # Remove rows with links
        chunk_cleaned = chunk_cleaned[~chunk_cleaned['body'].str.contains(r'http[s]?://\S+', na=False)]

        rows_after_cleaning += len(chunk_cleaned)
        print(f"Rows after cleaning: {len(chunk_cleaned)}")

        # Process the cleaned chunk
        chunk_processed = multiprocess_dataframe(chunk_cleaned, num_partitions=4)

        # (Optional) Save or accumulate chunks here

    # Summary
    print(f"\nTotal rows before cleaning: {total_rows}")
    print(f"Total rows after cleaning and filtering: {rows_after_cleaning}")

# Run the function
process_entire_dataset(
    "/scratch/general/vast/u1472278/parler_posts_comments.csv",
    chunk_size=50000
)
