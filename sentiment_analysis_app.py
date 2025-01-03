import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
import streamlit as st
import joblib
import os

# Constants
DATA_FILE = 'Reviews.csv'
MODEL_FILE = 'sentiment_model.pkl'

# Load and preprocess dataset
@st.cache_data
def load_and_preprocess_data(file_path):
    data = pd.read_csv(file_path)
    data['Review'] = data['Review'].str.lower()
    return data

# Train and save the model
@st.cache_resource
def train_and_save_model(data, model_file):
    X = data['Review']
    y = data['Liked']
    model = make_pipeline(TfidfVectorizer(), MultinomialNB())
    model.fit(X, y)
    joblib.dump(model, model_file)
    return model

# Load or train the model
@st.cache_resource
def load_model(model_file, data=None):
    if os.path.exists(model_file):
        return joblib.load(model_file)
    elif data is not None:
        return train_and_save_model(data, model_file)
    else:
        raise FileNotFoundError("Model not found and no data provided to train one.")

# Neutral phrases for fallback detection
NEUTRAL_PHRASES = ["ok", "fine", "average", "not bad", "so-so", "okay"]

# App header
def app_header():
    st.markdown("""
        <style>
        .header {
            font-size: 32px;
            color: #ffffff;
            text-align: center;
            background-color: #4CAF50;
            padding: 20px;
            border-radius: 10px;
        }
        .subheader {
            font-size: 20px;
            color: #ffffff;
            text-align: center;
            background-color: #008CBA;
            padding: 10px;
            border-radius: 10px;
            margin-top: 10px;
        }
        </style>
        <div class="header">Sentiment Analysis Web App</div>
        <div class="subheader">Analyze user reviews and predict sentiment in real-time!</div>
    """, unsafe_allow_html=True)

# App footer
def app_footer():
    st.markdown("""
        <style>
        .footer {
            font-size: 16px;
            color: #808080;
            text-align: center;
            margin-top: 50px;
        }
        </style>
        <div class="footer">
            Built with ❤️ using Python, Scikit-learn, and Streamlit
        </div>
    """, unsafe_allow_html=True)

# Main app
def main():
    app_header()

    # Load data and model
    st.sidebar.title("App Options")
    st.sidebar.info("Upload the dataset and train the model.")
    
    if not os.path.exists(DATA_FILE):
        st.error("Dataset file 'Reviews.csv' not found. Please upload it.")
        return
    
    data = load_and_preprocess_data(DATA_FILE)
    model = load_model(MODEL_FILE, data)
    
    # User input
    st.subheader("Enter Your Review")
    user_review = st.text_area("Write your review below:", placeholder="Type here...", height=150)

    if st.button("Analyze Sentiment"):
        if user_review.strip():
            if any(phrase in user_review.lower() for phrase in NEUTRAL_PHRASES):
                sentiment_text = "Neutral 😐"
                confidence = None  # Use None to indicate no confidence level for neutral
            else:
                sentiment = model.predict([user_review.lower()])[0]
                probabilities = model.predict_proba([user_review.lower()])[0]
                confidence = max(probabilities) * 100
                sentiment_text = "Positive 😊" if sentiment == 1 else "Negative 😞"

            # Display results
            st.success(f"Predicted Sentiment: **{sentiment_text}**")
            if confidence is not None:
                st.write(f"**Confidence Level:** {confidence:.2f}%")
            else:
                st.write("**Confidence Level:** N/A")
                
            if sentiment_text == "Positive 😊":
                st.balloons()
        else:
            st.warning("Please enter a review to analyze.")

    # Dataset exploration
    st.sidebar.subheader("Dataset Insights")
    if st.sidebar.checkbox("Show Sample Data"):
        st.subheader("Sample Data from Reviews Dataset")
        st.write(data.sample(5))
    
    # App footer
    app_footer()

# Entry point
if __name__ == "__main__":
    main()
