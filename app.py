import streamlit as st
import joblib
import os
import pandas as pd

MODEL_PATH = "spam_model.pkl"

# Load or notify
if not os.path.exists(MODEL_PATH):
    st.error("Model not found. Please run `train_model.py` first.")
    st.stop()

model = joblib.load(MODEL_PATH)

# UI
st.title("Spam vs. Ham Email Classifier")
st.write("Enter an email message to classify it.")

email = st.text_area("✉️ Email Text", height=200)

if st.button("🔍 Predict"):
    prediction = model.predict([email])[0]
    label = "Spam" if prediction else "Ham"
    st.success(f"🧠 Predicted: **{label}**")

    feedback = st.radio("Is the prediction correct?", ["Yes", "No"], index=0)
    if feedback == "No":
        correct_label = st.selectbox("Select the correct label:", ["Ham", "Spam"])
        if st.button(" Retrain with correction"):
            # Load past corrections if exist
            if os.path.exists("corrections.csv"):
                corrections = pd.read_csv("corrections.csv")
            else:
                corrections = pd.DataFrame(columns=["text", "label"])

            # Append and save correction
            new_entry = pd.DataFrame([[email, 0 if correct_label == "Ham" else 1]], columns=["text", "label"])
            corrections = pd.concat([corrections, new_entry], ignore_index=True)
            corrections.to_csv("corrections.csv", index=False)

            # Retrain model
            full_data = pd.read_csv("spam_data.csv")
            full_data['label'] = full_data['label'].map({'ham': 0, 'spam': 1})
            full_data = pd.concat([full_data, corrections], ignore_index=True)

            from sklearn.model_selection import train_test_split
            from sklearn.linear_model import LogisticRegression
            from sklearn.feature_extraction.text import CountVectorizer
            from sklearn.pipeline import Pipeline

            X_train, _, y_train, _ = train_test_split(full_data['text'], full_data['label'], test_size=0.2, random_state=42)

            new_model = Pipeline([
                ('vectorizer', CountVectorizer()),
                ('classifier', LogisticRegression())
            ])
            new_model.fit(X_train, y_train)
            joblib.dump(new_model, MODEL_PATH)
            st.success("Model updated with new correction.")
