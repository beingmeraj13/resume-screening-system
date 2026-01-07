AI-Based Resume Screening System

🔍 Project Overview
This project is an NLP-based Resume Screening System that automatically classifies resumes into job categories using machine learning.
It demonstrates an end-to-end pipeline from text preprocessing to model deployment using Flask.

🎯 Objectives
Automate resume screening using NLP
Convert unstructured resume text into ML-ready features
Train and evaluate multi-class classification models
Deploy the trained model as a web application

🧠 NLP & ML Pipeline

Resume text cleaning and normalization
Tokenization, stopword removal, and lemmatization
Feature extraction using TF-IDF (unigrams + bigrams)
Model training using Logistic Regression & Naive Bayes
Model evaluation using accuracy and class-wise metrics
Deployment using Flask for real-time prediction

📊 Dataset
Resume Dataset (Kaggle)
Contains resumes labeled by job category
Multi-class classification problem

⚙️ Tech Stack
Python | NLP | NLTK | TF-IDF | Scikit-learn | Logistic Regression | Flask | Git | GitHub

📈 Model Performance
Logistic Regression achieved ~85–90% accuracy
TF-IDF with n-grams provided strong text representation

🌐 Web Application
Paste resume text into web form
Predicts job category instantly
Uses same preprocessing pipeline as training

🚀 How to Run Locally
git clone https://github.com/yourusername/resume-screening-system.git
cd resume-screening-system
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python app.py


Open:
http://127.0.0.1:5000

🔮 Future Enhancements
Resume PDF upload
Skill matching score with job description
UI improvements
Cloud deployment