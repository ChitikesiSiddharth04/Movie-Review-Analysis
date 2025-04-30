# 🎬 IMDB Movie Review Sentiment Analysis

This project is a **Natural Language Processing (NLP)** application that classifies IMDB movie reviews into positive or negative sentiments using machine learning models.

## 📌 Overview

Using the **IMDB Movie Review Dataset**, this project builds and compares multiple Naive Bayes classifiers to perform sentiment analysis. It involves data preprocessing, feature extraction with Bag of Words (BoW), model training, and deployment using `pickle`.

## 🛠️ Technologies & Libraries

- Python 🐍
- Pandas & NumPy
- NLTK (stopwords, tokenization, stemming)
- Scikit-learn (CountVectorizer, Naive Bayes models)
- Regex
- Pickle (for model serialization)

## 📂 Dataset

- Dataset used: `IMDB Dataset.csv`
- Format: Two columns — `review` (text), `sentiment` (positive/negative)

## 📈 Model Workflow

1. **Preprocessing**:
    - HTML tag removal
    - Removal of special characters
    - Lowercasing text
    - Stopword removal
    - Word stemming (using `SnowballStemmer`)

2. **Feature Extraction**:
    - CountVectorizer with max 1000 features

3. **Model Training**:
    - Naive Bayes classifiers: `GaussianNB`, `MultinomialNB`, `BernoulliNB`

4. **Evaluation**:
    - Models compared using accuracy on test set

5. **Deployment**:
    - Best model (`BernoulliNB`) and BoW vocabulary are saved using `pickle`

## 🔍 Sample Prediction

A custom review can be passed through the same preprocessing pipeline, and the model can predict whether the sentiment is positive or negative.

## 📁 Files

- `IMDB Dataset.csv`: Input dataset
- `model1.pkl`: Trained `BernoulliNB` model
- `bow.pkl`: Pickled Bag of Words vocabulary
- `sentiment_analysis.py` (or Jupyter notebook): Code file

## 🧠 How to Run

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the script:
   ```bash
   python sentiment_analysis.py
   ```

3. Use the saved model for predictions:
   ```python
   import pickle
   model = pickle.load(open('model1.pkl', 'rb'))
   ```

## 💡 Credits

- IMDB Dataset: [Kaggle IMDB Dataset](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)
- NLTK & Scikit-learn community
