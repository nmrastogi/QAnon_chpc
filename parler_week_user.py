import pandas as pd

# Load data
df = pd.read_csv("/scratch/general/vast/u1472278/parler_huge.csv")

# Filter by date range
start_date = "2020-08-01"
# end_date = "2020-12-05"
filtered_df = df[(df['createdAtformatted'] >= start_date)]

# Print total unique users
total_users = filtered_df["creator"].nunique()
print(f"Total unique users in date range: {total_users}")

# Compute unique authors per week
unique_authors_per_week = filtered_df.groupby('year_week')['creator'].nunique()
unique_authors_df_parler = unique_authors_per_week.reset_index(name='creator')

# Save the output
unique_authors_df_parler.to_csv("unique_authors_per_week_parler_full_range.csv", index=False)
print("Saved filtered data and weekly unique authors.")

##end