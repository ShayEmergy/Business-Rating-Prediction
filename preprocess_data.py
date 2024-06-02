import json
import pandas as pd
import os

# Load data from JSON files
data_path = 'data/'
with open(os.path.join(data_path, 'yelp_business.json'), 'r') as file:
    business_data = json.load(file)

with open(os.path.join(data_path, 'yelp_checkin.json'), 'r') as file:
    checkin_data = json.load(file)

with open(os.path.join(data_path, 'yelp_data.json'), 'r') as file:
    data_data = json.load(file)

with open(os.path.join(data_path, 'yelp_photo.json'), 'r') as file:
    photo_data = json.load(file)

with open(os.path.join(data_path, 'yelp_review.json'), 'r') as file:
    review_data = json.load(file)

with open(os.path.join(data_path, 'yelp_tip.json'), 'r') as file:
    tip_data = json.load(file)

with open(os.path.join(data_path, 'yelp_user.json'), 'r') as file:
    user_data = json.load(file)

# Convert JSON data to pandas DataFrames
business_df = pd.DataFrame(business_data)
checkin_df = pd.DataFrame(checkin_data)
data_df = pd.DataFrame(data_data)
photo_df = pd.DataFrame(photo_data)
review_df = pd.DataFrame(review_data)
tip_df = pd.DataFrame(tip_data)
user_df = pd.DataFrame(user_data)

# Merge dataframes on the 'business_id' column
merged_df = business_df
for df in [checkin_df, data_df, photo_df, review_df, tip_df, user_df]:
    merged_df = pd.merge(merged_df, df, on='business_id', how='left', suffixes=('', '_dup'))

# Remove duplicate columns
merged_df = merged_df.loc[:, ~merged_df.columns.str.endswith('_dup')]

# Drop irrelevant or non-numeric columns
columns_to_drop = [
    'address', 'categories', 'city', 'hours', 'name', 'neighborhood',
    'postal_code', 'state', 'time', 'business_id', 'attributes'
]
merged_df = merged_df.drop(columns=columns_to_drop)

# Fill NaN values with 0
merged_df_filled = merged_df.fillna(0)

# Save preprocessed data
merged_df_filled.to_csv(os.path.join(data_path, 'preprocessed_data.csv'), index=False)
