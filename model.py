import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import pickle

# ===============================
# Download Stopwords
# ===============================
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Keep important negation words
negation_words = {"not", "no", "nor", "never"}
stop_words = stop_words - negation_words


# ===============================
# 1️⃣ Load Dataset
# ===============================
df = pd.read_csv(r"C:\Sentiment_Analysis_Project\dataset.csv.csv")

# Keep only required columns
df = df[['Rating', 'Review Text']]
df = df.dropna()

# ===============================
# 2️⃣ Extract Numeric Rating
# ===============================
df['Rating'] = df['Rating'].astype(str).str.extract(r'(\d)')
df = df.dropna(subset=['Rating'])
df['Rating'] = df['Rating'].astype(int)

# ===============================
# 3️⃣ Convert Rating to Sentiment
# ===============================
def convert_sentiment(rating):
    if rating >= 4:
        return "Positive"
    elif rating == 3:
        return "Neutral"
    else:
        return "Negative"

df['sentiment'] = df['Rating'].apply(convert_sentiment)

# 🔎 Print class distribution
print("\nClass Distribution:")
print(df['sentiment'].value_counts())

# ===============================
# 4️⃣ Clean Review Text
# ===============================
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
df['cleaned_review'] = df['Review Text'].apply(clean_text)

# ===============================
# 5️⃣ Features and Labels
# ===============================
X = df['cleaned_review']
y = df['sentiment']

# ===============================
# 6️⃣ Stratified Train-Test Split
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ===============================
# 7️⃣ TF-IDF Vectorization (Improved)
# ===============================
vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2)
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# ===============================
# 8️⃣ Logistic Regression (Manual Class Weight)
# ===============================
model = LogisticRegression(
    class_weight={
        "Negative": 1,
        "Neutral": 2,   # Boost Neutral importance
        "Positive": 1
    },
    max_iter=1000
)

model.fit(X_train_tfidf, y_train)

# ===============================
# 9️⃣ Evaluation
# ===============================
y_pred = model.predict(X_test_tfidf)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# ===============================
# 🔟 Save Model & Vectorizer
# ===============================
pickle.dump(model, open("sentiment_model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("\nModel and Vectorizer saved successfully!")
