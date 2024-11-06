# scripts/test.py

import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Load dataset
data = pd.read_csv('data/iris.csv')
X_test = data.drop('species', axis=1)
y_test = data['species']

# Load model
model = joblib.load("model.joblib")

# Test model
y_pred = model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)

# Log test accuracy to MLflow
mlflow.set_tracking_uri("http://your-mlflow-server.com")  # Change to your MLflow URI
mlflow.start_run()
mlflow.log_metric("test_accuracy", test_accuracy)
mlflow.end_run()
print("Testing complete. Test Accuracy:", test_accuracy)
