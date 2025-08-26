
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# 1. Load the dataset
df = pd.read_csv("hospital-data.csv")

# For this project, let\'s predict the \'Hospital Type\' based on other features.
# We will need to handle missing values and encode categorical variables.

# 2. Data Preprocessing and Cleaning
# Drop columns that are not useful for prediction or have too many missing values
df = df.drop(columns=["Address 1", "Address 2", "Address 3", "County", "Phone Number"])

# For simplicity, we will focus on a few key features and the target variable.
# Let\'s select a subset of columns for this example.
columns_to_use = ["State", "Hospital Type", "Hospital Ownership", "Emergency Services"]
df_subset = df[columns_to_use].copy()

# Drop rows with missing values in our selected subset
df_subset.dropna(inplace=True)

# Convert categorical variables to numerical using one-hot encoding
df_encoded = pd.get_dummies(df_subset, columns=["State", "Hospital Ownership", "Emergency Services"])

# Separate features (X) and target (y)
X = df_encoded.drop("Hospital Type", axis=1)
y = df_encoded["Hospital Type"]

# 3. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Train a classification model
# We will use a RandomForestClassifier for this task.
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 5. Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("--- Machine Learning Model Evaluation ---")
print(f"Accuracy: {accuracy:.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 6. Save the trained model
joblib.dump(model, "hospital_type_classifier.pkl")
print("\nTrained model saved to hospital_type_classifier.pkl")

print("\nPython Project 2 (Machine Learning) completed.")


