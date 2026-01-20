import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import pickle

# =============================
# 1. LOAD DATA
# =============================
df = pd.read_csv("./model/train (1).csv")  # path to dataset

# =============================
# 2. DATA PREPROCESSING
# =============================

# Select required features
features = ["Pclass", "Sex", "Age", "SibSp", "Fare"]
target = "Survived"

df = df[features + [target]]

# Handle missing values
df["Age"].fillna(df["Age"].median(), inplace=True)
df["Fare"].fillna(df["Fare"].median(), inplace=True)

# Encode categorical columns
label_encoder = LabelEncoder()
df["Sex"] = label_encoder.fit_transform(df["Sex"])  # male=1, female=0

# Split data
X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Feature scaling (recommended for Logistic Regression)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =============================
# 3. MODEL TRAINING
# =============================
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# =============================
# 4. MODEL EVALUATION
# =============================
y_pred = model.predict(X_test_scaled)
print("Classification Report:\n", classification_report(y_test, y_pred))

# =============================
# 5. SAVE MODEL TO DISK
# =============================
with open("titanic_survival_model.pkl", "wb") as file:
    pickle.dump((model, scaler, label_encoder), file)

print("Model saved as titanic_survival_model.pkl")

# =============================
# 6. DEMONSTRATE RELOADING
# =============================

with open("titanic_survival_model.pkl", "rb") as file:
    loaded_model, loaded_scaler, loaded_encoder = pickle.load(file)

# Example prediction input (Pclass, Sex, Age, SibSp, Fare)
sample = np.array([[3, 1, 22, 1, 7.25]])  # 3rd class, male, age 22
sample_scaled = loaded_scaler.transform(sample)

prediction = loaded_model.predict(sample_scaled)

print("Reloaded Model Prediction:", "Survived" if prediction[0] == 1 else "Did Not Survive")
