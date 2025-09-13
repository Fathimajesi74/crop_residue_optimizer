import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

# 📌 Load dataset
df = pd.read_csv("crop_residue_dataset_p.csv")

st.title("🌾 Crop Residue Management - Prediction App")

# Encode categorical columns
data = df.copy()
label_encoders = {}

for col in data.columns:
    if data[col].dtype == "object":
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col].astype(str))
        label_encoders[col] = le

# Define features and target
target_col = "Crop_Type"  # 🔹 change if needed
X = data.drop(target_col, axis=1)
y = data[target_col]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

st.subheader("📊 Model Evaluation")
st.write("Accuracy:", round(acc, 2))
st.text("Classification Report:")
st.text(classification_report(y_test, y_pred))

# Prediction section
st.subheader("🔮 Try a Prediction")
inputs = {}
for col in X.columns:
    if col in label_encoders:
        options = list(label_encoders[col].classes_)
        val = st.selectbox(f"{col}", options)
        inputs[col] = label_encoders[col].transform([val])[0]
    else:
        val = st.number_input(f"{col}", float(df[col].min()), float(df[col].max()), float(df[col].mean()))
        inputs[col] = val

if st.button("Predict"):
    input_df = pd.DataFrame([inputs])
    prediction = model.predict(input_df)[0]
    
    if target_col in label_encoders:
        prediction = label_encoders[target_col].inverse_transform([prediction])[0]
    
    st.success(f"✅ Predicted {target_col}: {prediction}")
