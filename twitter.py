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

# Dictionaries for category matching (as provided)
fusion = ["brother", "sister", "family", "motherland", "our blood", "fatherland", "sons", "daughters", "kin", "my people", "my race", "our people", "European race", "ancestry", "ancestor", "descendant", "fellow", "brethren", "comrades"]
violence = ["kill", "hang", "bomb", "shoot", "slaughter", "execute", "execution", "punish", "death penalty", "massacre", "destroy", "must attack", "must fight", "revenge", "retribution", "eradicate", "starve", "die", "torture", "behead", "burn", "bring death to", "give them hell", "weapon", "firearm", "assassinate", "gun", "rifle", "knife", "grenade", "brutal steps", "molotov", "jihaad", "jihad", "set fire", "revolution", "forcible overthrow", "flamethrowers", "M1-16", "ammonium nitrate"]
identification1 = ["\\bwe\\b", "\\bus\\b", "\\bour\\b", "\\bthey\\b", "\\bthem\\b", "\\btheir\\b"]
identification2 = ["\\bI\\b", "\\bme\\b", "\\bmy\\b", "\\byou\\b", "\\byour\\b"]
slurs = ["kike", "nigger", "negro", "dirty jew", "spic", "fag", "goyim", "golem", "the jew", "global jewry", "pajeet", "bitch", "whore"]
demonisation = ["traitor", "evil", "enemy", "corrupt", "vicious", "barbaric", "depraved", "vile", "puppets", "perversion", "blood libel", "pervert", "pedo", "crime", "cruel", "bloody", "genocidal", "sinful", "deceitful", "invader", "poison", "parasite", "menace", "brutal", "ruthless", "bloodsucking", "dirty", "deceptive", "treacherous", "poisonous", "oppressive", "oppressor", "shird", "unbeliever", "immoral", "jahili", "pollute", "demolish", "shake the foundations", "dar ul-harb", "arrogant", "mischievous", "criminal", "deceivers", "liars"]
dehumanisation = ["animal", "plague", "impure", "brute", "dog", "lower iq", "lower being", "inferior", "squalid", "parasitic", "parasite", "creature", "trash", "filth", "vermin", "spider", "devil", "monster", "beast", "reptile", "reptiloid", "femoid", "reptilian", "snake", "cockroach", "beneath human skin", "sub human", "anti-human", "disease", "savage", "infest", "breed", "locust", "monkey", "gorilla", "rat", "microbe", "satan", "cancer", "scum"]
existential_threat = ["subjected to", "coerced", "brainwashed", "exterminated", "brutalised", "raped", "terrorised", "ravaged", "extinction", "replacement", "genocide", "robbed", "subjugate", "make war upon my people", "destroy", "subvert", "overwhelmed", "under siege", "demographic siege", "disenfranchise", "assault", "kill us", "kill our", "kill my", "running out of time", "run out of time", "last chance", "enslavement", "enslaved", "suffer", "plunder", "condemned to death", "destruction of all mankind", "at the brink of", "endanger", "annihilation", "decay"]
conspiracy = ["betray", "betrayal", "sell", "sold", "collude", "conspire", "fake", "fraud", "corruption", "corrupt", "zog", "great replacement", "white genocide", "kalergi", "pedo elites", "NWO", "illuminati", "inside job", "Eurabia"]
inevitable_war1 = ["war", "battle", "fight", "jihad", "jihaad", "collapse", "conflict"]
inevitable_war2 = ["imminent", "inevitable", "looming", "start", "begin", "already", "heading for", "ongoing", "stage", "phase", "when", "has been", "likely", "predict", "expect", "will happen", "has begun", "current", "impending"]
violence_justification = ["pre-emptive", "defend", "protect", "self-defense", "self-defence", "forced to fight", "no longer ignore", "act of defense", "purified", "purify", "need for war", "need for violence", "need for jihad", "struggle is imposed", "natural struggle", "cannot co-exist"]
martyr = ["die in glory", "sacrifice", "knight", "martyr", "die selflessly", "protecting our people", "immortal", "preserve", "act of preservation", "defend the world of the Lord", "defending the work of the Lord", "stand guard", "standing guard", "the herald", "release mankind from", "free from", "freed from"]
violent_role_model1 = ["breivik", "tarrant", "hitler", "crusius", "rodger", "baillet", "earnest", "minassian", "mcveigh", "christchurch", "poway", "el paso"]
violent_role_model2 = ["hero", "role model", "saint", "inspire", "inspiration", "inspiring", "support", "influenced"]
hopelessness1 = ["democracy", "democratic", "peaceful", "political", "system", "politics", "dialogue", "passivity"]
hopelessness2 = ["meaningless", "weak", "fail", "end", "vanish", "man-made", "flawed", "jahili", "given up"]

dictionaries = {
    'fusion': fusion,
    'violence': violence,
    'identification1': identification1,
    'identification2': identification2,
    'slurs': slurs,
    'demonisation': demonisation,
    'dehumanisation': dehumanisation,
    'existential_threat': existential_threat,
    'conspiracy': conspiracy,
    'inevitable_war1': inevitable_war1,
    'inevitable_war2': inevitable_war2,
    'violence_justification': violence_justification,
    'martyr': martyr,
    'violent_role_model1': violent_role_model1,
    'violent_role_model2': violent_role_model2,
    'hopelessness1': hopelessness1,
    'hopelessness2': hopelessness2
}


# Pre-compile regex patterns for each dictionary to improve performance
compiled_patterns = {category: [re.compile(rf'\b{re.escape(word)}\b') for word in words_list]
                     for category, words_list in dictionaries.items()}

def clean_text(text):
    if not isinstance(text, str):
        text = ''
    words = text.split()
    cleaned_words = [word.lower() for word in words if word.lower() not in nltk_stopwords]
    return ' '.join(cleaned_words)

def count_categories(text):
    category_counts = {category: 0 for category in dictionaries}  # Initialize counts
    for category, patterns in compiled_patterns.items():
        for pattern in patterns:
            if pattern.search(text):
                category_counts[category] += 1
    return category_counts

def count_tokens(text):
    return len(text.split())

def process_chunk(chunk):
    chunk['tweet_text_cleaned'] = chunk['tweet_text'].apply(clean_text)
    chunk['tweet_text_category_counts'] = chunk['tweet_text_cleaned'].apply(count_categories)
    chunk['tweet_text_token_count'] = chunk['tweet_text_cleaned'].apply(count_tokens)
    return chunk

def multiprocess_dataframe(df, num_partitions=4):
    df_split = np.array_split(df, num_partitions)
    with Pool(num_partitions) as pool:
        df_chunks = pool.map(process_chunk, df_split)
    return pd.concat(df_chunks).reset_index(drop=True)

def process_data_in_chunks(file_path, chunk_size=50000, start_chunk=2345, partial_file='partial_results.txt', final_file='final_summary_tweet_text_final.txt'):
    final_category_counts = {category: 0 for category in dictionaries}
    final_token_count = 0
    chunk_count = start_chunk  # Start from the specified chunk number
    total_rows = 0
    rows_after_cleaning = 0

    # Open file for appending results
    with open(partial_file, 'a') as out_file:
        out_file.write("Continuing Chunk-wise Results\n")

    # Calculate the rows to skip based on the starting chunk
    rows_to_skip = start_chunk * chunk_size

    # Iterate over chunks, starting from the specified chunk
    for chunk in pd.read_csv(file_path, usecols=['tweet_text'], chunksize=chunk_size, skiprows=range(1, rows_to_skip + 1)):
        total_rows += len(chunk)
        print(f"Processing chunk {chunk_count}...")

        # Step 1: Clean the chunk (remove NaN or empty 'tweet_text')
        chunk_cleaned = chunk.dropna(subset=['tweet_text'], how='all')
        chunk_cleaned['tweet_text'] = chunk_cleaned['tweet_text'].astype(str)  # Convert to string to avoid errors
        chunk_cleaned = chunk_cleaned[(chunk_cleaned['tweet_text'].str.strip() != '')]

        rows_after_cleaning += len(chunk_cleaned)

        # Step 2: Remove rows with links in 'tweet_text'
        chunk_cleaned = chunk_cleaned[~chunk_cleaned['tweet_text'].str.contains(r'http[s]?://\S+', na=False)]

        # Step 3: Process the cleaned chunk using multiple processors
        chunk_processed = multiprocess_dataframe(chunk_cleaned, num_partitions=4)

        # Save partial results after each chunk is processed
        total_tokens = chunk_processed['tweet_text_token_count'].sum()
        combined_category_counts_total = chunk_processed['tweet_text_category_counts'].apply(pd.Series).sum()

        # Ensure combined_category_counts_total is a Series with all categories
        if not isinstance(combined_category_counts_total, pd.Series):
            combined_category_counts_total = pd.Series({category: 0 for category in final_category_counts})

        # Accumulate total tokens and category counts
        final_token_count += total_tokens
        final_category_counts = {category: final_category_counts[category] + combined_category_counts_total.get(category, 0) for category in final_category_counts}

        # Append chunk results to the partial file
        with open(partial_file, 'a') as out_file:
            out_file.write(f"\nResults for chunk {chunk_count}:\n")
            out_file.write(f"Total number of tokens: {total_tokens}\n")
            out_file.write(f"Category counts:\n{combined_category_counts_total}\n")

        chunk_count += 1  # Increment the chunk count

    print(f"Total rows processed: {total_rows}")
    print(f"Rows after cleaning (going for tokenization): {rows_after_cleaning}")

    # Save the cumulative final results
    with open(final_file, 'a') as final_out_file:
        final_out_file.write("\nFinal Cumulative Results:\n")
        final_out_file.write(f"Total tokens: {final_token_count}\n")
        final_out_file.write(f"Final category counts:\n{final_category_counts}\n")

# Execute the function from chunk 2345 onward
processed_df = process_data_in_chunks(
    "/scratch/general/vast/u1472278/tweets.csv",
    chunk_size=50000,
    start_chunk=2345,
    partial_file='partial_results.txt',
    final_file='final_summary_tweet_text_final.txt'
)
