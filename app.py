import nltk
nltk.download('stopwords')

import streamlit as st
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# download stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# clean text function
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

# load dataset
df = pd.read_csv("reviews.csv")

# preprocess
df['cleaned'] = df['text_'].apply(clean_text)

# vectorize
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['cleaned'])
y = df['label']

# train model
model = LogisticRegression()
model.fit(X, y)

# ---------------- UI ----------------

st.title("🧠 Fake Review Detection System")

st.write("Enter a product review below:")

user_input = st.text_area("✍️ Review Text")

if st.button("Analyze"):
    cleaned = clean_text(user_input)
    vec = vectorizer.transform([cleaned])
    
    prediction = model.predict(vec)
    prob = model.predict_proba(vec)

    st.subheader("🔍 Result")

    # Prediction
    if prediction[0] == "CG":
        st.success("✅ Genuine Review")
    else:
        st.error("❌ Fake Review")

    # Trust score
    score = round(max(prob[0]) * 100, 2)

    st.progress(score / 100)
    st.write(f"Trust Score: {score}%")

    # Confidence level
    if score > 80:
        st.info("High confidence prediction")
    elif score > 60:
        st.warning("Moderate confidence")
    else:
        st.error("Low confidence")
    


st.set_page_config(page_title="Fake Review Detector", layout="centered")

st.markdown("<h3 style='font-style: italic; font-weight: normal;'>AI Review Authenticity Analyzer</h3>", unsafe_allow_html=True)
st.markdown("Detect whether a review is genuine or fake using Machine Learning")

if st.button("Try Sample"):
    user_input = "This product is amazing and worth buying"
    cleaned = clean_text(user_input)
    vec = vectorizer.transform([cleaned])
    
    prediction = model.predict(vec)
    prob = model.predict_proba(vec)

    score = round(max(prob[0]) * 100, 2)

    st.write(user_input)
    st.write(f"Prediction: {prediction[0]}")
    st.write(f"Trust Score: {score}%")

if 'score' in locals():
    if score > 80:
        st.info("High confidence prediction")
    elif score > 60:
        st.warning("Moderate confidence")
    else:
        st.error("Low confidence")