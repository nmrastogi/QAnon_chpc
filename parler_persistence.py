import pandas as pd

# Load data
df = pd.read_csv("/scratch/general/vast/u1472278/parler_huge.csv")

# Filter by date range
start_date = "2020-09-28"
end_date = "2020-12-05"
filtered_df = df[(df['createdAtformatted'] >= start_date) & (df['createdAtformatted'] <= end_date)]

# Group by creator and year_week
grouped_df = filtered_df.groupby(['creator', 'year_week']).size().reset_index(name='count')

# Count unique weeks per creator
unique_weeks_per_creator = grouped_df.groupby('creator')['year_week'].nunique().reset_index(name='unique_year_week_counts')

# Save outputs
unique_weeks_per_creator.to_csv("parler_persistence.csv", index=False)

print("creator_unique_week_counts")
