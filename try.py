import pandas as pd
import re
import nltk
from nltk.corpus import stopwords

# Download NLTK stopwords if you haven't already
nltk.download('stopwords')

# Load NLTK English stopwords
nltk_stopwords = set(stopwords.words('english'))

# Load your full CSV file
df = pd.read_csv('datasets/dotwin_comments.csv')

# Clean the dataframe: Remove NaN values, empty rows, and rows with links
df_cleaned = df.dropna(subset=['content'])
df_cleaned = df_cleaned[df_cleaned['content'].str.strip() != '']
df_cleaned = df_cleaned[~df_cleaned['content'].str.contains(r'http[s]?://\S+', na=False)]
df_cleaned = df_cleaned.reset_index(drop=True)

# Dictionaries
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

# Function to clean text: remove stop words and lower the text
def clean_text(text):
    words = text.split()
    cleaned_words = [word.lower() for word in words if word.lower() not in nltk_stopwords]
    return ' '.join(cleaned_words)

# Pre-compile regex patterns for each dictionary
compiled_patterns = {category: [re.compile(rf'\b{re.escape(word)}\b') for word in words_list]
                     for category, words_list in dictionaries.items()}

# Function to count occurrences from each dictionary in the text
def count_categories(text):
    category_counts = {category: 0 for category in dictionaries}
    for category, patterns in compiled_patterns.items():
        for pattern in patterns:
            if pattern.search(text):
                category_counts[category] += 1
    return category_counts

# Clean the text in the 'content' column only
df_cleaned['content_cleaned'] = df_cleaned['content'].apply(clean_text)

# Counter to print progress every 50,000 rows
progress_interval = 50000
total_rows = len(df_cleaned)

# Count occurrences of each category in the cleaned text, print progress every 50,000 rows
combined_category_counts_total = pd.Series(dtype='int')

for index, row in df_cleaned.iterrows():
    # Count categories in the cleaned text for each row
    row_counts = count_categories(row['content_cleaned'])
    combined_category_counts_total = combined_category_counts_total.add(pd.Series(row_counts), fill_value=0)
    
    # Print progress every 50,000 rows
    if (index + 1) % progress_interval == 0:
        print(f"Processed {index + 1} rows out of {total_rows}...")

# Save the results to a file
with open('category_counts_results_full.txt', 'w') as file:
    file.write("Total category counts for the 'content' column across all rows:\n")
    file.write(str(combined_category_counts_total))

# Final print after all rows are processed
print(f"Processed all {total_rows} rows.")
print("Category counts for all rows have been saved to 'category_counts_results_full.txt'.")
