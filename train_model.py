import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle
import os

# Load preprocessed data
data_path = 'data/'
merged_df_filled = pd.read_csv(os.path.join(data_path, 'preprocessed_data.csv'))

# Calculate the correlation matrix
correlation_matrix = merged_df_filled.corr()

# Extract the correlation values with the target variable 'stars'
correlation_with_stars = correlation_matrix['stars']

# Identify features with the highest positive and negative correlations
positive_correlations = correlation_with_stars[correlation_with_stars > 0].sort_values(ascending=False)
negative_correlations = correlation_with_stars[correlation_with_stars < 0].sort_values()

# Select features with significant correlations (arbitrarily setting a threshold of 0.05 for this example)
significant_features = correlation_with_stars[abs(correlation_with_stars) > 0.05].index.tolist()

# Remove 'stars' from the list of features to avoid including the target variable as a feature
significant_features.remove('stars')

# Create a DataFrame of just the target column (Yelp ratings stars)
target_df = merged_df_filled[['stars']]

# Create a DataFrame of the selected significant features
features_df = merged_df_filled[significant_features]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features_df, target_df, test_size=0.2, random_state=1)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Save the model
with open('results/model.pkl', 'wb') as file:
    pickle.dump(model, file)

# Save the test data
X_test.to_csv(os.path.join(data_path, 'X_test.csv'), index=False)
y_test.to_csv(os.path.join(data_path, 'y_test.csv'), index=False)
