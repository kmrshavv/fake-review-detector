import pandas as pd

print("Code is running...")

df = pd.read_csv("reviews.csv")

print(df.head())
print("\nColumns:", df.columns)
print("\nShape:", df.shape)

import re
import nltk
from nltk.corpus import stopwords

# download once
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)   # remove symbols
    words = text.split()
    words = [w for w in words if w not in stop_words]  # remove stopwords
    return " ".join(words)

# apply cleaning on text_ column
df['cleaned'] = df['text_'].apply(clean_text)

print("\nCleaned Data:")
print(df[['text_', 'cleaned']].head())

from sklearn.feature_extraction.text import TfidfVectorizer

# convert text to numbers
vectorizer = TfidfVectorizer(max_features=5000)

X = vectorizer.fit_transform(df['cleaned'])
y = df['label']   # target (fake or real)

print("\nTF-IDF Shape:", X.shape)

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# train model
lr_model = LogisticRegression()
lr_model.fit(X_train, y_train)

# accuracy
accuracy = lr_model.score(X_test, y_test)
print("\nModel Accuracy:", accuracy)

sample = ["This product is amazing and worth buying"]

sample_clean = clean_text(sample[0])
sample_vec = vectorizer.transform([sample_clean])

prediction = lr_model.predict(sample_vec)

print("\nTest Review:", sample[0])
print("Prediction:", prediction[0])

from sklearn.metrics import classification_report, confusion_matrix

y_pred = lr_model.predict(X_test)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(n_estimators=200, random_state=42)
rf_model.fit(X_train, y_train)

model = rf_model

accuracy = model.score(X_test, y_test)
print("\nNew Accuracy:", accuracy)

import numpy as np

def get_trust_score(text):
    cleaned = clean_text(text)
    vec = vectorizer.transform([cleaned])
    
    prob = model.predict_proba(vec)[0]
    pred = model.predict(vec)[0]
    
    confidence = np.max(prob)
    
    # smarter logic
    if pred == "CG":
        trust = confidence * 100
    else:
        trust = (1 - confidence) * 100
        
    return round(trust, 2), pred


sample = "I have been using this product for 6 months and the performance is consistently good"

score, label = get_trust_score(sample)

print("\nPrediction:", label)
print("Trust Score:", score)

def interpret_score(score):
    if score > 80:
        return "Highly Trustworthy"
    elif score > 60:
        return "Moderately Trustworthy"
    elif score > 40:
        return "Suspicious"
    else:
        return "Likely Fake"


# CALL FUNCTION OUTSIDE
score, label = get_trust_score(sample)
status = interpret_score(score)

print("\nPrediction:", label)
print("Trust Score:", score)
print("Status:", status)