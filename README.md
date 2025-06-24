# 🎬 IMDB Movie Review Sentiment Analysis Web App

This project is a complete, deployable web application that performs sentiment analysis on movie reviews. It uses a machine learning model trained on the IMDB 50k Movie Review dataset to classify reviews as either "Positive" or "Negative". The frontend is built with a clean, modern, and dynamic UI that uses JavaScript to provide a seamless user experience.

![Demo](https://i.imgur.com/your-demo-image.gif)  <!-- Replace with a GIF of your app! -->

## ✨ Features

- **Accurate Sentiment Analysis:** Utilizes a `Bernoulli Naive Bayes` classifier trained on 50,000 movie reviews.
- **Dynamic Frontend:** Single-page application experience built with vanilla JavaScript, HTML, and CSS. No page reloads are needed for analysis.
- **Responsive Design:** A clean and modern UI that looks great on all screen sizes.
- **Ready for Deployment:** Includes configuration for easy, free deployment on platforms like Render.

## 🛠️ Tech Stack

- **Backend:** Python, Flask
- **Machine Learning:** Scikit-learn, Pandas, NLTK
- **Frontend:** HTML, CSS, JavaScript
- **Deployment:** Gunicorn, Render

## 📂 File Structure

```
.
├── app.py              # Main Flask application
├── MRA.py              # Script to train the ML model
├── model1.pkl          # Pickled trained BernoulliNB model
├── bow.pkl             # Pickled CountVectorizer
├── requirements.txt    # Python dependencies
├── templates/
│   └── index.html      # Frontend HTML, CSS, and JS
├── .gitignore          # Files to be ignored by Git
├── render-build.sh     # Build script for deployment
└── README.md           # This file
```

---

## 🚀 Running Locally

To run this project on your own machine, follow these steps.

### 1. Prerequisites

- Python 3.8+
- Git

### 2. Clone & Setup

```bash
# Clone the repository
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name

# Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate

# Install the dependencies
pip install -r requirements.txt
```

### 3. Get the Dataset & Train the Model

The pre-trained model files (`model1.pkl` and `bow.pkl`) are already included in this repository. However, if you wish to retrain the model yourself, you'll need the original dataset.

1.  **Download the dataset** from [Kaggle: IMDB Dataset of 50K Movie Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews).
2.  Place the `IMDB Dataset.csv` file in the root of the project directory.
3.  **Run the training script:**
    ```bash
    python MRA.py
    ```
    This will regenerate `model1.pkl` and `bow.pkl` based on the dataset.

### 4. Run the Web App

```bash
# Start the Flask development server
python app.py
```

Open your browser and navigate to `http://127.0.0.1:5000` to use the application.

---

## ☁️ Deployment to Render

This application is ready to be deployed for free on [Render](https://render.com/).

### 1. Create a GitHub Repository

Push your project code to a new repository on GitHub.

### 2. Deploy on Render

1.  **Create a new Render Account** or log in.
2.  On your dashboard, click **New +** and select **Web Service**.
3.  **Connect your GitHub repository**.
4.  Fill in the service details:
    -   **Name:** Give your app a unique name (e.g., `movie-review-analysis-app`).
    -   **Region:** Choose a region near you.
    -   **Branch:** `main` (or your default branch).
    -   **Root Directory:** Leave it blank.
    -   **Runtime:** `Python 3`.
    -   **Build Command:** `./render-build.sh`
    -   **Start Command:** `gunicorn app:app`
5.  Click **Create Web Service**.

Render will automatically build and deploy your application. Once it's live, you'll get a public URL you can share with anyone!
