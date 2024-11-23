from flask import Flask, render_template, request, jsonify
import json
import pickle
import numpy as np
from scipy.sparse import hstack
from textblob import TextBlob
import re
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

# Ensure NLTK resources are available
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

app = Flask(__name__)

# Load Pretrained Model and TF-IDF Vectorizer
with open("./models/xgb_review_checker_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("./models/tfidf_vectorizer.pkl", "rb") as vectorizer_file:
    tfidf = pickle.load(vectorizer_file)

# Load Users and Books Data
with open("users.json", "r") as users_file:
    users = json.load(users_file)

with open("books.json", "r") as books_file:
    books = json.load(books_file)

# Initialize Lemmatizer and Stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Helper functions
def preprocess_text(text):
    """Preprocess the review text by cleaning and lemmatizing."""
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)
    text = re.sub(r'\W', ' ', text)
    text = ' '.join([lemmatizer.lemmatize(word.lower()) for word in text.split() if word.lower() not in stop_words])
    return text

def calculate_helpfulness_ratio(helpfulness):
    """Calculate the helpfulness ratio from the 'helpfulness' string (e.g., '2/3')."""
    try:
        if isinstance(helpfulness, str) and '/' in helpfulness:
            numerator, denominator = map(int, helpfulness.split('/'))
            return numerator / max(denominator, 1)
    except ValueError:
        pass
    return 0

def get_sentiment(text):
    """Compute sentiment polarity and subjectivity using TextBlob."""
    blob = TextBlob(text)
    return blob.sentiment.polarity, blob.sentiment.subjectivity

@app.route("/")
def index():
    """Render the main form for input."""
    return render_template("index.html", users=users, books=books)

@app.route("/predict", methods=["POST"])
def predict():
    """Predict whether a review is real or fake based on the input data."""
    try:
        # Get form data
        user_id = request.form.get("user_id")
        book_title = request.form.get("book_title")
        review_text = request.form.get("review_text")

        # Validate input
        if not user_id or not book_title or not review_text:
            return jsonify({"error": "Missing input data"}), 400

        # Find selected book and user data
        selected_book = next((book for book in books if book["Title"] == book_title), None)
        selected_user = next((user for user in users if user["user_id"] == user_id), None)

        if not selected_book or not selected_user:
            return jsonify({"error": "Invalid user or book selection"}), 400

        # Process user and book features
        user_helpfulness_ratio = calculate_helpfulness_ratio(selected_user["helpfulness"])
        numeric_features = np.array([
            user_helpfulness_ratio,
            selected_book.get("avg_helpfulness_ratio", 0),
            selected_book.get("avg_rating", 0),
            selected_book.get("total_reviews", 0),
            selected_book.get("avg_word_count", 0),
            selected_book.get("avg_sentiment_polarity", 0),
            selected_book.get("avg_sentiment_subjectivity", 0),
            len(review_text.split()),  # word_count
            len([word for word in review_text.split() if word in {"good", "great", "excellent", "amazing", "love", "awesome"}]),  # positive_words
            len([word for word in review_text.split() if word in {"bad", "poor", "terrible", "worst", "hate"}]),  # negative_words
            get_sentiment(review_text)[0],  # sentiment_polarity
            get_sentiment(review_text)[1]   # sentiment_subjectivity
        ]).reshape(1, -1)

        # Process review text
        preprocessed_text = preprocess_text(review_text)
        text_features = tfidf.transform([preprocessed_text])

        # Ensure TF-IDF features have the expected shape
        expected_text_features = 500  # Matches TF-IDF max_features during training
        if text_features.shape[1] < expected_text_features:
            padding = np.zeros((1, expected_text_features - text_features.shape[1]))
            text_features = hstack([text_features, padding])

        # Combine features
        combined_features = hstack([numeric_features, text_features])

        # DEBUG: Ensure combined features match the model input shape
        if combined_features.shape[1] != 512:
            return jsonify({"error": f"Feature shape mismatch, expected 512 but got {combined_features.shape[1]}"}), 400

        # Perform prediction
        prediction = model.predict(combined_features)
        result = "Real" if prediction[0] == 1 else "Fake"

        return jsonify({"result": result})

    except Exception as e:
        return jsonify({"error": f"Error in prediction: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(debug=True)
