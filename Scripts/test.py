import joblib
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris  # Example dataset


import os
print("Current working directory:", os.getcwd())

# Load the trained model (Assuming it's saved as 'saved_model.pkl')
model = joblib.load('saved_model.pkl')

# Example: Using Iris dataset for testing (you can replace it with your own dataset)
data = load_iris()
X = data.data
y = data.target

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

# Print the results
print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"Classification Report:\n{report}")
