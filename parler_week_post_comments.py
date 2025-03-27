import pandas as pd

# Load data
df = pd.read_csv("/scratch/general/vast/u1472278/parler_huge.csv")

# Filter by date range
start_date = "2020-09-28"
end_date = "2020-12-05"
filtered_df = df[(df['createdAtformatted'] >= start_date) & (df['createdAtformatted'] <= end_date)]

# Separate posts
po = filtered_df[filtered_df["datatype"] == "posts"]
week_counts = po.groupby('year_week').size()
week_counts_df_parler = week_counts.reset_index(name='row_count')
print("Total post rows in range:", week_counts_df_parler["row_count"].sum())

# Save post weekly counts to CSV
week_counts_df_parler.to_csv("parler_posts_week.csv", index=False)

# Separate comments
co = filtered_df[filtered_df["datatype"] == "comments"]
week_counts_co = co.groupby('year_week').size()
week_counts_df_1_parler = week_counts_co.reset_index(name='row_count')
print("Total comment rows in range:", week_counts_df_1_parler["row_count"].sum())

# Save comment weekly counts to CSV
week_counts_df_1_parler.to_csv("parler_comments_week.csv", index=False)
