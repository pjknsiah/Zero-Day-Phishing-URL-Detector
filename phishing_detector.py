import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score

# --- STEP 1: LOAD DATA ---
print("Loading URL data...")
df = pd.read_csv('urldata.csv')

# The dataset usually has 'url' and 'label' (bad/good)
# Let's clean it up just in case
df = df.dropna()

print(f"Total URLs: {len(df)}")
print("Example Malicious URL:", df[df['label'] == 'bad']['url'].iloc[0])

# --- STEP 2: TOKENIZATION (The 'Research' Secret Sauce) ---
# Standard NLP tokenizers split by spaces. URLs don't have spaces.
# We need a custom tokenizer to split by '/', '-', '.' to find hidden patterns.
def make_tokens(f):
    # Split by slash, then by dot, then by hyphen
    tokens_by_slash = str(f).split('/')
    total_tokens = []
    for i in tokens_by_slash:
        tokens = str(i).split('-')
        tokens_by_dot = []
        for j in range(0,len(tokens)):
            temp_tokens = str(tokens[j]).split('.')
            tokens_by_dot = tokens_by_dot + temp_tokens
        total_tokens = total_tokens + tokens + tokens_by_dot
    # Remove 'com', 'www' because they are everywhere and useless for detection
    total_tokens = list(set(total_tokens))
    if 'com' in total_tokens:
        total_tokens.remove('com')
    return total_tokens

print("\nVectorizing URLs (This turns text into math)...")
# TF-IDF: Weighs "rare" words (like 'login-secure-update') heavily, and common words lightly.
vectorizer = TfidfVectorizer(tokenizer=make_tokens, token_pattern=None)
X = vectorizer.fit_transform(df['url'])
y = df['label']

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- STEP 3: TRAIN (Logistic Regression) ---
# Logistic Regression is the industry standard for high-speed classification
print("Training Model...")
logit = LogisticRegression(max_iter=1000)
logit.fit(X_train, y_train)

# --- STEP 4: EVALUATE ---
print("\nEvaluating...")
y_pred = logit.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n✅ Model Accuracy: {accuracy * 100:.2f}%")
print("\n--- Classification Report ---")
print(classification_report(y_test, y_pred))

# --- STEP 5: REAL-TIME TEST ---
# Let's test it on a fake phishing link right now
fake_phish = ["www.secure-login-update-apple.com/account-verify"]
fake_vector = vectorizer.transform(fake_phish)
prediction = logit.predict(fake_vector)
print(f"\nTesting on fake link '{fake_phish[0]}': {prediction[0]}")