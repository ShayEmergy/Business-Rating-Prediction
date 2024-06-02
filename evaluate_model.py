import pandas as pd
from sklearn.metrics import r2_score
import pickle
import os

# Load the model
with open('results/model.pkl', 'rb') as file:
    model = pickle.load(file)

# Load the test data
data_path = 'data/'
X_test = pd.read_csv(os.path.join(data_path, 'X_test.csv'))
y_test = pd.read_csv(os.path.join(data_path, 'y_test.csv'))

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
r2 = r2_score(y_test, y_pred)
print("\nR^2 Score:", r2)

# Save evaluation results
with open('results/evaluation.txt', 'w') as file:
    file.write(f"R^2 Score: {r2}\n")

# Select the first 5 rows from the test set for predictions
sample_data = X_test.iloc[:5]

# Use the trained model to make predictions on the sample data
sample_predictions = model.predict(sample_data)

# Save the predictions and the actual values
with open('results/evaluation.txt', 'a') as file:
    file.write(f"\nSample Predictions: {sample_predictions}\n")
    file.write(f"\nActual Values: {y_test.iloc[:5].values}\n")
