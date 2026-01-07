from flask import Flask, render_template, request
import pickle
import pandas as pd
import sys
import os

# Allow imports from src
sys.path.append(os.path.abspath("."))

from src.text_preprocess import clean_resume_text

app = Flask(__name__)

# Load trained model & TF-IDF
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("tfidf.pkl", "rb") as f:
    tfidf = pickle.load(f)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None

    if request.method == "POST":
        resume_text = request.form["resume_text"]

        # Preprocess
        cleaned_text = clean_resume_text(resume_text)

        # Vectorize
        vectorized_text = tfidf.transform([cleaned_text])

        # Predict
        prediction = model.predict(vectorized_text)[0]

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
