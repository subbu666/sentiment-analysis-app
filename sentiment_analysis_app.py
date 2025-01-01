import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
import streamlit as st
import joblib
import os

# Define constants
DATA_FILE = 'Reviews.csv'
MODEL_FILE = 'sentiment_model.pkl'

# Load and preprocess dataset
def load_and_preprocess_data(file_path):
    data = pd.read_csv(file_path)
    data['Review'] = data['Review'].str.lower()
    return data

# Train and save the model
def train_and_save_model(data, model_file):
    X = data['Review']
    y = data['Liked']
    model = make_pipeline(CountVectorizer(), MultinomialNB())
    model.fit(X, y)
    joblib.dump(model, model_file)

# Load the model
def load_model(model_file):
    if os.path.exists(model_file):
        return joblib.load(model_file)
    else:
        raise FileNotFoundError(f"{model_file} not found.")

# Streamlit App
st.title("Sentiment Analysis App")
st.write("Enter a review, and the app will predict whether it's positive or negative.")

# Check if the model exists, else train it
if not os.path.exists(MODEL_FILE):
    st.write("Training the model...")
    data = load_and_preprocess_data(DATA_FILE)
    train_and_save_model(data, MODEL_FILE)

# Load the trained model
model = load_model(MODEL_FILE)

# Input from the user
user_review = st.text_area("Enter your review:")

if st.button("Submit"):
    if user_review.strip():
        sentiment = model.predict([user_review.lower()])[0]
        sentiment_text = "Positive" if sentiment == 1 else "Negative"
        st.success(f"Predicted Sentiment: {sentiment_text}")
    else:
        st.error("Please enter a valid review.")
