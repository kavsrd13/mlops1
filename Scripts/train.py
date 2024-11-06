# scripts/train.py


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset
data = pd.read_csv('data/iris.csv')
X = data.drop('species', axis=1)
y = data['species']

# Split data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# Validation
y_pred = model.predict(X_val)
accuracy = accuracy_score(y_val, y_pred)
model_filename = 'saved_model.pkl'

# Ensure the model is saved in the correct directory
model_path = os.path.join(os.getcwd(), model_filename)  # Saves in the current directory

# Save the model
joblib.dump(model, model_path)

print("Training complete. Accuracy:", accuracy)
