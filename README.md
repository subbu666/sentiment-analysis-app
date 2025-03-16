# 🌟 Sentiment Analysis Web App


## 📌 Overview
This **Sentiment Analysis Web App** analyzes user reviews and classifies them as **Positive 😊, Negative 😞, or Neutral 😐**. It is built with **Machine Learning** and **Streamlit**, providing an interactive interface for real-time sentiment prediction.

---
## ✨ Features
✅ **Preprocessing**: Cleans and processes text data
✅ **ML Model**: Uses **TF-IDF Vectorization** + **Naïve Bayes**
✅ **Real-time Prediction**: Get instant sentiment results
✅ **Confidence Score**: View model confidence level
✅ **Dataset Insights**: Explore sample reviews
✅ **User-friendly UI**: Modern, sleek design with animations

---
## 🔧 Technologies Used
| Technology | Purpose |
|------------|---------|
| **Python** 🐍 | Programming Language |
| **Streamlit** 🎨 | Web Framework |
| **Scikit-learn** 🤖 | Machine Learning |
| **Pandas** 🏗 | Data Handling |
| **Joblib** 💾 | Model Persistence |

---
## 📂 Dataset
The dataset used for training the model is stored in **`Reviews.csv`**, containing user reviews and sentiment labels.

---
## 🚀 Installation & Setup
### Clone the Repository
```bash
git clone https://github.com/yourusername/sentiment-analysis-app.git
cd sentiment-analysis-app
```
### Install Dependencies
```bash
pip install -r requirements.txt
```
### Run the Web App
```bash
streamlit run sentiment_analysis_app.py
```
🎉 **Enjoy real-time sentiment analysis on your own data!**

---
## 🔥 Usage Guide
1️⃣ **Enter a review** in the input box.
2️⃣ Click **"Analyze Sentiment"** to classify the text.
3️⃣ View the **sentiment result** & **confidence score**.
4️⃣ Explore **sample dataset insights** (optional).

---
## 🏆 Model Details
📌 **Technique:** TF-IDF + Multinomial Naïve Bayes  
📌 **Training:** Uses `Reviews.csv` to train the model  
📌 **Model Storage:** If `sentiment_model.pkl` exists, it loads the pre-trained model; otherwise, a new model is trained.  
📌 **Neutral Sentiments:** Identified using predefined phrases.

---
## 🚀 Future Enhancements
🔹 Upgrade to **Deep Learning** (LSTMs, Transformers)  
🔹 Add **Multilingual Sentiment Analysis**  
🔹 Enhance UI/UX with animations & visualizations  


---
🚀 **Built with ❤️ using Python, Scikit-learn, and Streamlit!**
