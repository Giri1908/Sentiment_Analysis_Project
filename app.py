import streamlit as st
import pickle
import re
import nltk
import numpy as np
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
import io

# -------------------------------
# Download stopwords
# -------------------------------
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

negation_words = {"not", "no", "nor", "never"}
stop_words = stop_words - negation_words

# -------------------------------
# Load model and vectorizer
# -------------------------------
model = pickle.load(open("sentiment_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# -------------------------------
# Text cleaning function
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
# PDF Generation Function
# -------------------------------
def generate_pdf(review, prediction, prob_dict):

    buffer = io.BytesIO()

    doc = SimpleDocTemplate(buffer)

    styles = getSampleStyleSheet()

    elements = []

    title = Paragraph(
        "Product Review Sentiment Report",
        styles["Title"]
    )
    elements.append(title)

    elements.append(Spacer(1, 12))

    review_text = Paragraph(
        f"<b>Review:</b> {review}",
        styles["BodyText"]
    )
    elements.append(review_text)

    elements.append(Spacer(1, 12))

    sentiment_text = Paragraph(
        f"<b>Predicted Sentiment:</b> {prediction}",
        styles["BodyText"]
    )
    elements.append(sentiment_text)

    elements.append(Spacer(1, 12))

    confidence_text = Paragraph(
        f"<b>Confidence Scores:</b><br/>"
        f"Negative: {prob_dict['Negative']:.2f}<br/>"
        f"Neutral: {prob_dict['Neutral']:.2f}<br/>"
        f"Positive: {prob_dict['Positive']:.2f}",
        styles["BodyText"]
    )

    elements.append(confidence_text)

    doc.build(elements)

    buffer.seek(0)

    return buffer

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
# Product Category Selector
# -------------------------------
st.subheader("📦 Select Product Category")

product_category = st.selectbox(
    "Choose Product Type:",
    [
        "Laptop",
        "Mobile Phone",
        "Headphones",
        "Smartwatch",
        "Camera"
    ]
)

# -------------------------------
# Display Images
# -------------------------------
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
review = st.text_area(
    "✍️ Enter your review here:"
)

# -------------------------------
# Prediction Button
# -------------------------------
if st.button("Predict Sentiment"):

    if review.strip() == "":
        st.warning("Please enter a review first.")

    else:

        # Clean and vectorize text
        cleaned = clean_text(review)

        vectorized = vectorizer.transform([cleaned])

        # Predict sentiment
        prediction = model.predict(vectorized)[0]

        probabilities = model.predict_proba(vectorized)[0]

        # -----------------------
        # Show Sentiment Result
        # -----------------------
        if prediction == "Positive":
            st.success(f"Sentiment: {prediction}")

        elif prediction == "Negative":
            st.error(f"Sentiment: {prediction}")

        else:
            st.info(f"Sentiment: {prediction}")

        # -----------------------
        # Probability Dictionary
        # -----------------------
        prob_dict = {
            "Negative": float(probabilities[0]),
            "Neutral": float(probabilities[1]),
            "Positive": float(probabilities[2])
        }

        # -----------------------
        # Colored Bar Chart
        # -----------------------
        st.subheader("📊 Prediction Confidence")

        labels = ["Negative", "Neutral", "Positive"]

        values = [
            prob_dict["Negative"],
            prob_dict["Neutral"],
            prob_dict["Positive"]
        ]

        colors = ["red", "blue", "green"]

        fig, ax = plt.subplots()

        ax.bar(labels, values, color=colors)

        ax.set_xlabel("Sentiment")
        ax.set_ylabel("Confidence Score")
        ax.set_title("Sentiment Prediction Confidence")

        st.pyplot(fig)

        # -----------------------
        # Show Raw Scores
        # -----------------------
        st.write("Confidence Scores:")
        st.write(prob_dict)

        # -----------------------
        # Generate PDF Report
        # -----------------------
        pdf_file = generate_pdf(
            review,
            prediction,
            prob_dict
        )

        st.success("Report generated successfully!")

        # -----------------------
        # Download Button
        # -----------------------
        st.download_button(
            label="📄 Download Final Report",
            data=pdf_file,
            file_name="sentiment_report.pdf",
            mime="application/pdf"
        )
