# Phishing URL Detection System (Lexical Analysis)

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![ML](https://img.shields.io/badge/Model-Logistic%20Regression-green)
![Status](https://img.shields.io/badge/Accuracy-96.1%25-brightgreen)

A machine learning-based security tool designed to detect malicious URLs in real-time based on **lexical analysis** rather than traditional blacklists. By utilizing **TF-IDF vectorization** and a **custom tokenizer**, this system identifies phishing attempts by analyzing URL structure, subdomain patterns, and path tokens.

## 🚀 Project Overview

Traditional phishing detection relies heavily on blacklists (databases of known bad sites). However, phishing domains are often short-lived (hours or days), making blacklists ineffective against **Zero-Day attacks**.

This project solves that problem by treating URL detection as a **Natural Language Processing (NLP)** task. It breaks down a URL into its constituent parts to find semantic patterns associated with malicious intent (e.g., `secure-login`, `update-account` in subdomains).

**Key Features:**
* **Custom Tokenizer:** Splits URLs by non-standard delimiters (`/`, `-`, `.`) to extract meaningful keywords often hidden in long phishing URLs.
* **TF-IDF Vectorization:** Weighs rare, suspicious keywords heavily while ignoring common terms (like `www` or `com`).
* **Lightweight Inference:** Uses **Logistic Regression** for millisecond-latency prediction, suitable for real-time browser integration.

## 📊 Key Results

* **Dataset:** 400,000+ URLs (Legitimate vs. Phishing).
* **Model:** Logistic Regression (optimized for speed/accuracy trade-off).
* **Accuracy:** **96.1%** on the test set.
* **Precision/Recall:** High recall on phishing class, minimizing the risk of missing a dangerous link.

### Real-World Test Cases
 The model successfully identifies synthetic "zero-day" URLs that are not in any database:
* `google.com-security-check.xy/login` $\to$ **PHISHING** (Detected via hyphenated subdomain pattern)
* `www.amazon-orders-update.com` $\to$ **PHISHING** (Detected via suspicious keyword combination)

## 🛠️ Installation & Setup

### Prerequisites
* Python 3.8+
* pip

### 1. Clone the Repository
```bash
git clone [https://github.com/YOUR_USERNAME/phishing-url-detection.git](https://github.com/YOUR_USERNAME/phishing-url-detection.git)
cd phishing-url-detection

### 2. Install Dependencies
```bash
pip install pandas numpy scikit-learn

### 3. Get the Dataset
```bash
Download urldata.csv (or use the provided script to fetch it) and place it in the root directory.