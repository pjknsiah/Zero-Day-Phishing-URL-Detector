# Phishing URL Detection System (Lexical Analysis)

A machine learning-based security tool designed to detect malicious URLs in real-time based on **lexical analysis** rather than traditional blacklists. By utilizing **TF-IDF vectorization** and a **custom tokenizer**, this system identifies phishing attempts by analyzing URL structure, subdomain patterns, and path tokens.

## Project Overview

Traditional phishing detection relies heavily on blacklists (databases of known bad sites). However, phishing domains are often short-lived (hours or days), making blacklists ineffective against **Zero-Day attacks**.

This project solves that problem by treating URL detection as a **Natural Language Processing (NLP)** task. It breaks down a URL into its constituent parts to find semantic patterns associated with malicious intent (e.g., `secure-login`, `update-account` in subdomains).

**Key Features:**
* **Custom Tokenizer:** Splits URLs by non-standard delimiters (`/`, `-`, `.`) to extract meaningful keywords often hidden in long phishing URLs.
* **TF-IDF Vectorization:** Weighs rare, suspicious keywords heavily while ignoring common terms (like `www` or `com`).
* **Lightweight Inference:** Uses **Logistic Regression** for millisecond-latency prediction, suitable for real-time browser integration.

## Key Results

* **Dataset:** 400,000+ URLs (Legitimate vs. Phishing).
* **Model:** Logistic Regression (optimized for speed/accuracy trade-off).
* **Accuracy:** **96.1%** on the test set.
* **Precision/Recall:** High recall on phishing class, minimizing the risk of missing a dangerous link.

### Real-World Test Cases
 The model successfully identifies synthetic "zero-day" URLs that are not in any database:
* `google.com-security-check.xy/login` $\to$ **PHISHING** (Detected via hyphenated subdomain pattern)
* `www.amazon-orders-update.com` $\to$ **PHISHING** (Detected via suspicious keyword combination)

## Installation & Setup

### Prerequisites
* Python 3.8+
* pip

### 1. Clone the Repository
```bash
git clone https://github.com/pjknsiah/Zero-Day-Phishing-URL-Detector.git
cd Zero-Day-Phishing-URL-Detector
```

### 2. Install Dependencies
```bash
pip install pandas numpy scikit-learn
```

### 3. Prepare the Dataset
The project requires a dataset named `urldata.csv`. Ensure this file is in the root directory.
(Note: The repository includes a sample dataset, or you can download a comprehensive one from Kaggle).

## Usage

Run the detector script directly:

```bash
python phishing_detector.py
```

The script will:
1. Load `urldata.csv`.
2. Train the Logistic Regression model.
3. Output the accuracy and a classification report.
4. Test on a sample synthetic phishing URL.

## Deep Dive: How It Works

1.  **Tokenization**: We split URLs not just by `/` but also by special characters like `-`, `.`, and `_`. This exposes hidden semantic words that attackers use (e.g., `secure`, `update`, `banking`) which are often mashed together in subdomains.
2.  **Vectorization (TF-IDF)**:
    *   **TF (Term Frequency)**: How often a word appears in a URL.
    *   **IDF (Inverse Document Frequency)**: Reduces the weight of common terms (`com`, `http`) and boosts rare, suspicious terms.
3.  **Model (Logistic Regression)**: A linear classifier that learns weights for each token. It is extremely fast, making it ideal for checking URLs in real-time (e.g., inside a browser extension or proxy).

## Future Improvements

*   **GUI / Web Interface**: Build a Flask/Django frontend for easy user interaction.
*   **Browser Extension**: Port the detection logic to a Chrome/Firefox extension.
*   **Model serialization**: Save the trained model (`pickle`/`joblib`) to avoid retraining on every run.
*   **Deep Learning**: Explore CNNs/LSTMs for character-level patterns.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.
