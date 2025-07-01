# train_pipeline.py

import pandas as pd
import numpy as np
import pickle
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import classification_report, mean_squared_error

# --- Utility Functions ---
def clean_text(text):
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# --- Load and Preprocess Data ---
def load_and_preprocess(csv_path):
    df = pd.read_csv(csv_path)
    df.dropna(subset=['review', 'rating'], inplace=True)
    df['review_clean'] = df['review'].apply(clean_text)
    df['sentiment'] = df['rating'].apply(lambda x: 'positive' if x >= 3 else 'negative')
    return df

# --- Train Models ---
def train_models(df):
    X = df['review_clean']
    y_sentiment = df['sentiment']
    y_rating = df['rating']

    # Vectorize
    vectorizer = TfidfVectorizer(max_features=5000)
    X_vect = vectorizer.fit_transform(X)

    # Split
    X_train, X_test, y_train_sent, y_test_sent = train_test_split(X_vect, y_sentiment, test_size=0.2, random_state=42)
    _, _, y_train_rat, y_test_rat = train_test_split(X_vect, y_rating, test_size=0.2, random_state=42)

    # Train Sentiment Model
    sentiment_model = LogisticRegression(max_iter=1000)
    sentiment_model.fit(X_train, y_train_sent)
    y_pred_sent = sentiment_model.predict(X_test)
    print("Sentiment Classification Report:\n", classification_report(y_test_sent, y_pred_sent))

    # Train Rating Model
    rating_model = LinearRegression()
    rating_model.fit(X_train, y_train_rat)
    y_pred_rat = rating_model.predict(X_test)
    print("Rating RMSE:", np.sqrt(mean_squared_error(y_test_rat, y_pred_rat)))

    # Save Models
    with open("sentiment_model.pkl", "wb") as f:
        pickle.dump(sentiment_model, f)
    with open("rating_model.pkl", "wb") as f:
        pickle.dump(rating_model, f)
    with open("tfidf_vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)

    return sentiment_model, rating_model, vectorizer

# --- Entry Point ---
if __name__ == "__main__":
    df = load_and_preprocess("chatgpt_reviews.csv")  # Ensure this CSV is available
    train_models(df)
