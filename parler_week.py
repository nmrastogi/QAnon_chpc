import pandas as pd

# Load data
df = pd.read_csv("../largedata/new_range/parler_huge.csv").head(1000000)

# Filter by date range
start_date = "2020-09-28"
end_date = "2020-12-05"
filtered_df = df[(df['createdAtformatted'] >= start_date) & (df['createdAtformatted'] <= end_date)]

# Print total unique users
total_users = filtered_df["creator"].nunique()
print(f"Total unique users in date range: {total_users}")

# Compute unique authors per week
unique_authors_per_week = filtered_df.groupby('year_week')['creator'].nunique()
unique_authors_df_parler = unique_authors_per_week.reset_index(name='creator')

# Save the output
unique_authors_df_parler.to_csv("unique_authors_per_week_parler_.csv", index=False)
print("Saved filtered data and weekly unique authors.")
