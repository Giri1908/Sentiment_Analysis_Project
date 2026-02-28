import streamlit as st
import pickle
import re
import nltk
import numpy as np
from nltk.corpus import stopwords

# -------------------------------
# Download stopwords (first time only)
# -------------------------------
nltk.download('stopwords')

# Load stopwords
stop_words = set(stopwords.words('english'))
negation_words = {"not", "no", "nor", "never"}
stop_words = stop_words - negation_words

# -------------------------------
# Load saved model and vectorizer
# -------------------------------
model = pickle.load(open("sentiment_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# -------------------------------
# Text cleaning function (UNCHANGED)
# -------------------------------
def clean_text(text):
    text = re.sub('[^a-zA-Z]', ' ', str(text))
    text = text.lower()
    words = text.split()

    processed_words = []
    i = 0

    while i < len(words):
        if words[i] in {"not", "no", "never"} and i + 1 < len(words):
            combined_word = words[i] + "_" + words[i+1]
            processed_words.append(combined_word)
            i += 2
        else:
            if words[i] not in stop_words:
                processed_words.append(words[i])
            i += 1

    return ' '.join(processed_words)

# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(
    page_title="Sentiment Analyzer",
    page_icon="🛍️",
    layout="centered"
)

st.title("🛍️ Product Review Sentiment Analyzer")
st.write("Analyze customer reviews using Machine Learning")

st.markdown("---")

# -------------------------------
# NEW: Product Category Selector
# -------------------------------
st.subheader("📦 Select Product Category")

product_category = st.selectbox(
    "Choose Product Type:",
    ["Laptop", "Mobile Phone", "Headphones", "Smartwatch", "Camera"]
)

# -------------------------------
# NEW: Display Product Images
# -------------------------------
# Make sure you create an "images" folder in your project

if product_category == "Laptop":
    st.image("images/laptop.jpg", use_container_width=True)

elif product_category == "Mobile Phone":
    st.image("images/mobile.jpg", use_container_width=True)

elif product_category == "Headphones":
    st.image("images/headphones.jpg", use_container_width=True)

elif product_category == "Smartwatch":
    st.image("images/smartwatch.jpg", use_container_width=True)

elif product_category == "Camera":
    st.image("images/camera.jpg", use_container_width=True)

st.markdown("---")

# -------------------------------
# Review Input
# -------------------------------
review = st.text_area("✍️ Enter your review here:")

if st.button("Predict Sentiment"):

    if review.strip() == "":
        st.warning("Please enter a review first.")
    else:
        cleaned = clean_text(review)
        vectorized = vectorizer.transform([cleaned])

        prediction = model.predict(vectorized)[0]
        probabilities = model.predict_proba(vectorized)[0]

        # Show sentiment result
        if prediction == "Positive":
            st.success(f"Sentiment: {prediction}")
        elif prediction == "Negative":
            st.error(f"Sentiment: {prediction}")
        else:
            st.info(f"Sentiment: {prediction}")

        # Show probability chart
        st.subheader("📊 Prediction Confidence")

        prob_dict = {
            "Negative": float(probabilities[0]),
            "Neutral": float(probabilities[1]),
            "Positive": float(probabilities[2])
        }

        st.bar_chart(prob_dict)

        # Show raw confidence values
        st.write("Confidence Scores:")
        st.write(prob_dict)
