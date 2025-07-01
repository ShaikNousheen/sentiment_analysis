import streamlit as st
import pickle
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import pipeline
import torch

# Load TF-IDF-based models
sentiment_model = pickle.load(open("sentiment_model.pkl", "rb"))
rating_model = pickle.load(open("rating_model.pkl", "rb"))
tfidf_vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))

# Load BERT components
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
bert_pipeline = pipeline("sentiment-analysis", model=bert_model, tokenizer=bert_tokenizer)

# Clean text function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# LDA topic modeling
def extract_topics(texts, n_topics=3, n_words=5):
    count_vect = CountVectorizer(stop_words='english', max_df=0.9, min_df=2)
    dtm = count_vect.fit_transform(texts)
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    lda.fit(dtm)
    topics = []
    for i, topic in enumerate(lda.components_):
        words = [count_vect.get_feature_names_out()[index] for index in topic.argsort()[-n_words:]]
        topics.append(f"Topic {i+1}: " + ", ".join(words))
    return topics

# Streamlit UI
st.set_page_config(page_title="ChatGPT Review Analyzer", layout="centered")
st.title("📝 ChatGPT Review Analyzer")
st.markdown("Analyze ChatGPT reviews for sentiment, ratings, and deeper insights.")

# User Input
user_input = st.text_area("Enter a review:")

model_choice = st.radio("Choose model for Sentiment Prediction:", ("Traditional (TF-IDF)", "BERT Transformer"))

if st.button("Analyze Review"):
    if not user_input.strip():
        st.warning("Please enter a review.")
    else:
        cleaned = clean_text(user_input)
        vect_text = tfidf_vectorizer.transform([cleaned])
        
        # Rating prediction using traditional model
        rating = rating_model.predict(vect_text)[0]

        # Sentiment prediction
        if model_choice == "Traditional (TF-IDF)":
            sentiment = sentiment_model.predict(vect_text)[0]
        else:
            bert_result = bert_pipeline(user_input)[0]
            sentiment = bert_result['label'].lower()  # 'positive' or 'negative'

        # Display Results
        st.markdown(f"### Sentiment: `{sentiment.capitalize()}`")
        st.markdown(f"### Predicted Rating: ⭐ `{round(float(rating), 2)}` / 5")

        # Emoji Response
        if sentiment == 'positive':
            st.success("Thanks for the positive feedback! 😊")
        elif sentiment == 'negative':
            st.error("We'll try to improve based on your review. 😔")
        else:
            st.info("Thanks for your feedback. 🤔")

        # Topic Modeling
        st.markdown("#### 🔍 Extracted Topics (LDA):")
        topics = extract_topics([cleaned])
        for t in topics:
            st.write(t)

# Optional: Time-Series Sentiment Trend (example only)
st.markdown("---")
st.subheader("📈 Time-Series Sentiment Trend (Demo Data)")
sample_data = pd.read_csv("sample_reviews_timeseries.csv")  # Preprocessed CSV with 'review', 'date', 'sentiment'

# Visualization
if not sample_data.empty:
    sample_data['date'] = pd.to_datetime(sample_data['date'])
    trend_data = sample_data.groupby([sample_data['date'].dt.to_period('M'), 'sentiment']).size().unstack().fillna(0)
    trend_data.index = trend_data.index.to_timestamp()
    st.line_chart(trend_data)
