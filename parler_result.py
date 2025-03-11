import re
import pandas as pd

# Define the file path
file_path = 'path_to_your_file.txt'

# Initialize a dictionary to store cumulative counts for each category and total tokens
cumulative_counts = {
    "total_tokens": 0,
    "fusion": 0,
    "violence": 0,
    "identification1": 0,
    "identification2": 0,
    "slurs": 0,
    "demonisation": 0,
    "dehumanisation": 0,
    "existential_threat": 0,
    "conspiracy": 0,
    "inevitable_war1": 0,
    "inevitable_war2": 0,
    "violence_justification": 0,
    "martyr": 0,
    "violent_role_model1": 0,
    "violent_role_model2": 0,
    "hopelessness1": 0,
    "hopelessness2": 0
}

# Regular expressions to match relevant lines
token_pattern = re.compile(r"Total number of tokens:\s+(\d+)")
category_pattern = re.compile(r"(\w+)\s+(\d+)")

# Parse the file and aggregate counts
with open(file_path, 'r') as file:
    for line in file:
        # Check for total tokens line
        token_match = token_pattern.search(line)
        if token_match:
            cumulative_counts["total_tokens"] += int(token_match.group(1))
        
        # Check for category lines
        category_match = category_pattern.search(line)
        if category_match:
            category, count = category_match.groups()
            if category in cumulative_counts:
                cumulative_counts[category] += int(count)

# Calculate the percentage for each category based on total tokens
total_tokens = cumulative_counts["total_tokens"]
percentages = {category: (count / total_tokens) * 100 for category, count in cumulative_counts.items() if category != "total_tokens"}

# Convert cumulative counts and percentages to DataFrames
df_cumulative_counts = pd.DataFrame([cumulative_counts])
df_percentages = pd.DataFrame([percentages])

# Display the cumulative counts and percentages
print("Aggregated Counts:\n", df_cumulative_counts)
print("\nCategory Percentages by Total Tokens:\n", df_percentages)
