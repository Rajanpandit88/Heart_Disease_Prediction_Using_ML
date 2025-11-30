📌 README.md – Heart Disease Prediction Using Machine Learning
❤️ Heart Disease Prediction Using Machine Learning

This project aims to predict the likelihood of heart disease in patients using multiple Machine Learning classification models. It uses a structured dataset containing patient health attributes such as age, cholesterol levels, chest-pain type, resting blood pressure, and more.
The goal is to provide a data-driven decision support system for early detection of heart disease.

🚀 Features

Data preprocessing and cleaning
Exploratory Data Analysis (EDA)
Implementation of multiple ML models:
Logistic Regression
Random Forest Classifier
Neural Network (MLP Classifier)
Decision Tree
KNN
Model comparison based on accuracy, precision, recall, and F1-score
Final deployed model saved as .pkl
Jupyter Notebook workflow
CSV dataset included

📂 Project Structure
Heart-Disease-Prediction/
│
├── Heart_Disease_Prediction_Using_Machine_Learning.ipynb
├── heart.csv
├── random_forest_model.pkl
├── neural_network_model.pkl
├── README.md
└── requirements.txt (optional)

🧠 Machine Learning Workflow

Load and clean dataset
Handle missing values
Feature scaling & encoding
Train/Test split
Train multiple ML models
Evaluate performance
Select best model and save with Pickle

📊 Results

The Random Forest and Neural Network models performed with the highest accuracy.
Final models are exported as .pkl files for deployment.

🗂️ Dataset

File: heart.csv

Contains 303 rows and 14 features related to medical attributes.

Common columns include:

age, sex, cp, trestbps, chol, thalach, ca, thal, target, etc.

🧪 How to Run the Project

Install required libraries:
pip install numpy pandas scikit-learn matplotlib

Run the Jupyter Notebook:

jupyter notebook


Open: Heart_Disease_Prediction_Using_Machine_Learning.ipynb

Run all cells sequentially to train and evaluate models.

📦 Model Deployment

The trained models are stored in:

random_forest_model.pkl
neural_network_model.pkl

These can be imported like:

import pickle

model = pickle.load(open("random_forest_model.pkl", "rb"))
prediction = model.predict([input_data])

📁 Future Improvements

Build a web interface using Flask
Add API endpoint
Improve model accuracy using hyperparameter tuning
Implement cross-validation

👤 Author

Rajan Pandit
Machine Learning & Data Science Enthusiast
