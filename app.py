from flask import Flask, request, render_template, jsonify
import re
import nltk
import joblib
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer

app = Flask(__name__)

# Load ML model and vectorizer
ml_model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# VADER analyzer
vader_analyzer = SentimentIntensityAnalyzer()

# Preprocess for ML model
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    words = [w for w in words if w not in stop_words]
    return ' '.join(words)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    review = data.get('review', '')

    if not review.strip():
        return jsonify({'error': 'No review provided'}), 400

    word_count = len(review.strip().split())
    review_clean = review.lower().strip()

    # Custom neutral keyword override
    neutral_keywords = ["okay", "fine", "average", "not bad", "meh", "alright", "nothing special"]
    if review_clean in neutral_keywords:
        sentiment = "neutral"
        print(f"Matched neutral keyword: {review_clean}")
    elif word_count <= 5:
        # VADER for short reviews
        scores = vader_analyzer.polarity_scores(review)
        compound = scores['compound']
        print(f"Compound score for VADER: {compound} | Review: {review}")

        if compound >= 0.3:
            sentiment = "positive"
        elif compound <= -0.3:
            sentiment = "negative"
        else:
            sentiment = "neutral"
    else:
        # ML model for longer reviews
        cleaned = preprocess_text(review)
        vec = vectorizer.transform([cleaned])
        sentiment = ml_model.predict(vec)[0]

    return jsonify({"sentiment": sentiment})


if __name__ == '__main__':
    app.run(debug=True)

