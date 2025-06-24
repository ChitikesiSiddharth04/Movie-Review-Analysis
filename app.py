import numpy as np
import pickle
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
from flask import Flask, request, render_template, jsonify

# Initialize Flask app
app = Flask(__name__)

# Load the trained model and CountVectorizer
model = pickle.load(open('model1.pkl', 'rb'))
cv = pickle.load(open('bow.pkl', 'rb'))

# Preprocessing functions from MRA.py
def clean(text):
    cleaned = re.compile(r'<.*?>')
    return re.sub(cleaned, '', text)

def is_special(text):
    rem = ''
    for i in text:
        if i.isalnum():
            rem = rem + i
        else:
            rem = rem + ' '
    return rem

def to_lower(text):
    return text.lower()

def rem_stopwords(text):
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text)
    return [w for w in words if w not in stop_words]

def stem_txt(text):
    ss = SnowballStemmer('english')
    return " ".join([ss.stem(w) for w in text])

def preprocess_review(review):
    # This function chains all the preprocessing steps
    f1 = clean(review)
    f2 = is_special(f1)
    f3 = to_lower(f2)
    f4 = rem_stopwords(f3)
    return stem_txt(f4)

# Main route
@app.route('/')
def home():
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        review = request.form['review']
        
        # Preprocess, transform, and predict
        processed_review = preprocess_review(review)
        review_vector = cv.transform([processed_review]).toarray()
        y_pred = model.predict(review_vector)
        
        sentiment = 'Positive' if y_pred[0] == 1 else 'Negative'
        
        # Return a JSON response
        return jsonify({'review': review, 'sentiment': sentiment})
    
    # This part will not be reached for POST requests but is good practice
    return jsonify({'error': 'Invalid request method'}), 405

if __name__ == '__main__':
    app.run(debug=True) 