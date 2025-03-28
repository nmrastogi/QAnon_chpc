import pandas as pd
import re
from nltk.corpus import stopwords
import nltk
from multiprocessing import Pool
import numpy as np

# Download NLTK stopwords
nltk.download('stopwords')

# Load stopwords
nltk_stopwords = set(stopwords.words('english'))

# Date range filter
start_date = "2020-09-28"
end_date = "2020-12-05"

# Dictionary categories
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
    'fusion': fusion, 'violence': violence, 'identification1': identification1, 'identification2': identification2,
    'slurs': slurs, 'demonisation': demonisation, 'dehumanisation': dehumanisation,
    'existential_threat': existential_threat, 'conspiracy': conspiracy,
    'inevitable_war1': inevitable_war1, 'inevitable_war2': inevitable_war2,
    'violence_justification': violence_justification, 'martyr': martyr,
    'violent_role_model1': violent_role_model1, 'violent_role_model2': violent_role_model2,
    'hopelessness1': hopelessness1, 'hopelessness2': hopelessness2
}

compiled_patterns = {
    category: [re.compile(rf'\b{re.escape(word)}\b', re.IGNORECASE) for word in words]
    for category, words in dictionaries.items()
}

def clean_text(text):
    if not isinstance(text, str): return ''
    words = text.split()
    return ' '.join([word.lower() for word in words if word.lower() not in nltk_stopwords])

def count_categories(text):
    return {cat: sum(bool(p.search(text)) for p in patterns) for cat, patterns in compiled_patterns.items()}

def count_tokens(text):
    return len(text.split())

def process_chunk(chunk):
    chunk['body_cleaned'] = chunk['body'].apply(clean_text)
    chunk['body_category_counts'] = chunk['body_cleaned'].apply(count_categories)
    chunk['body_token_count'] = chunk['body_cleaned'].apply(count_tokens)
    return chunk

def multiprocess_dataframe(df, num_partitions=4):
    df_split = np.array_split(df, num_partitions)
    with Pool(num_partitions) as pool:
        df_chunks = pool.map(process_chunk, df_split)
    return pd.concat(df_chunks).reset_index(drop=True)

def process_data_in_chunks(file_path, chunk_size=50000, partial_file='same_range_partial_results_parler.txt', final_file='same_range_final_summary_parler.txt'):
    final_category_counts = {category: 0 for category in dictionaries}
    final_token_count = 0
    chunk_count = 0
    total_rows = 0
    rows_after_cleaning = 0

    with open(partial_file, 'w') as out_file:
        out_file.write("Chunk-wise Results\n")

    for chunk in pd.read_csv(file_path, usecols=['body', 'createdAtformatted'], chunksize=chunk_size):
        chunk_count += 1
        print(f"\nProcessing chunk {chunk_count}...")
        total_rows += len(chunk)

        # Date filter
        chunk = chunk[(chunk['createdAtformatted'] >= start_date) & (chunk['createdAtformatted'] <= end_date)]
        print(f"Rows after date filtering: {len(chunk)}")

        chunk = chunk.dropna(subset=['body'])
        chunk['body'] = chunk['body'].astype(str)
        chunk = chunk[chunk['body'].str.strip() != '']
        chunk = chunk[~chunk['body'].str.contains(r'http[s]?://\S+', na=False)]

        if chunk.empty:
            continue

        rows_after_cleaning += len(chunk)

        chunk_processed = multiprocess_dataframe(chunk, num_partitions=4)

        total_tokens = chunk_processed['body_token_count'].sum()
        combined_counts = chunk_processed['body_category_counts'].apply(pd.Series).sum()

        final_token_count += total_tokens
        final_category_counts = {
            cat: final_category_counts[cat] + combined_counts.get(cat, 0)
            for cat in final_category_counts
        }

        with open(partial_file, 'a') as out_file:
            out_file.write(f"\nChunk {chunk_count}:\n")
            out_file.write(f"Tokens: {total_tokens}\n")
            out_file.write(f"Category counts:\n{combined_counts}\n")

    print(f"\nTotal rows processed: {total_rows}")
    print(f"Rows after cleaning: {rows_after_cleaning}")

    with open(final_file, 'w') as final_out_file:
        final_out_file.write("Final Results:\n")
        final_out_file.write(f"Total tokens: {final_token_count}\n")
        final_out_file.write(f"Final category counts:\n{final_category_counts}\n")

# Run the function
process_data_in_chunks(
    "/scratch/general/vast/u1472278/parler_posts_comments.csv"
)
